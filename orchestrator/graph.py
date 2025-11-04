from langgraph.graph import StateGraph, END
from models.state import State
from agents.search import SearchAgent
from agents.acquisition import AcquisitionAgent
from agents.filtering_agent import FilteringAgent
from agents.processing_agent import ProcessingAgent
from agents.query_optimization_agent import QueryOptimizationAgent
from agents.deduplication_agent import DeduplicationAgent
from agents.fallback_refinement_agent import FallbackRefinementAgent
from agents.analysis import AnalysisAgent
from utils.logging_decorator import log_state_transition
from typing import Dict, Any
from loguru import logger
import json
import asyncio

try:
    from src.necthrall_lite.services.synthesis_service import SynthesisAgent
except ImportError:
    # Fallback if import fails
    SynthesisAgent = None


def should_refine_query(state: State) -> str:
    """
    Routing function: Determine if fallback refinement is needed.

    Triggers refinement if:
    - search_quality.passed is False (insufficient papers or low relevance)
    - AND refinement_count < 2 (max 2 refinement attempts)

    Args:
        state: Current LangGraph State

    Returns:
        "refine" if refinement needed, "continue" otherwise
    """
    search_quality = state.search_quality or {}
    passed = search_quality.get("passed", True)  # Default to True if missing
    refinement_count = state.refinement_count

    if not passed and refinement_count < 2:
        logger.info(
            f"Search quality check failed (attempt {refinement_count + 1}/2): "
            f"{search_quality.get('reason', 'Unknown reason')}"
        )
        return "refine"

    if refinement_count >= 2:
        logger.warning(
            "Max refinement attempts (2) reached, continuing with available papers"
        )

    return "continue"


def synthesis_node(state: State) -> State:
    """
    Synthesis node: Generate citation-grounded answers from retrieved passages and credibility scores.

    Updates state with synthesized_answer, citations, and consensus_estimate.
    Handles synthesis failures gracefully by logging errors and setting error messages in state.

    Args:
        state: Current State with query, relevant_passages, credibility_scores, contradictions

    Returns:
        Updated State with synthesis results
    """
    logger.info("Starting synthesis node execution")

    try:
        if SynthesisAgent is None:
            raise ImportError("SynthesisAgent not available")

        # Initialize agent
        agent = SynthesisAgent()

        # Prepare input data
        # Note: ProcessingAgent populates top_passages (legacy field), not relevant_passages
        passages = []
        for passage in state.top_passages:
            passages.append(
                {
                    "content": passage.content,
                    "paper_id": passage.paper_id,
                    "section": passage.section,
                }
            )

        credibility_scores = []
        for score in state.credibility_scores:
            credibility_scores.append(
                {
                    "paper_id": score.paper_id,
                    "score": score.score,
                    "tier": score.tier,
                    "rationale": score.rationale,
                }
            )

        contradictions = []
        for contra in state.contradictions:
            contradictions.append(
                {
                    "topic": contra.topic,
                    "claim_1": {
                        "paper_id": contra.claim_1.paper_id,
                        "text": contra.claim_1.text,
                    },
                    "claim_2": {
                        "paper_id": contra.claim_2.paper_id,
                        "text": contra.claim_2.text,
                    },
                    "severity": contra.severity,
                }
            )

        # Call synthesis with retry logic: up to max_attempts attempts.
        max_attempts = 2
        attempt = 0
        result = None
        last_validation = None

        while attempt < max_attempts:
            attempt += 1
            logger.info(f"Synthesis attempt {attempt}/{max_attempts}")

            try:
                result = asyncio.run(
                    agent.synthesize(
                        query=state.original_query,
                        passages=passages,
                        credibility_scores=credibility_scores,
                        contradictions=contradictions if contradictions else None,
                    )
                )
            except Exception as e:
                logger.warning(
                    json.dumps(
                        {
                            "event": "synthesis_call_failure",
                            "attempt": attempt,
                            "error": str(e),
                        }
                    )
                )
                # If we've exhausted attempts, raise to outer handler
                if attempt >= max_attempts:
                    raise
                else:
                    # continue to next attempt
                    continue

            # Validate citations for hallucinations or invalid indices
            try:
                # Prefer the real validation implementation from the service module
                try:
                    from src.necthrall_lite.services.synthesis_service import (
                        SynthesisAgent as _RealSynthesisAgent,
                    )

                    validator = _RealSynthesisAgent.validate_citations
                except Exception:
                    # Fallback to whatever is available on the (possibly patched) SynthesisAgent
                    validator = getattr(SynthesisAgent, "validate_citations", None)

                if validator is None:
                    raise RuntimeError("No citation validator available")

                last_validation = validator(result.answer, passages)
            except Exception as e:
                # Validation should not crash pipeline; treat as failed validation
                logger.warning(
                    json.dumps({"event": "citation_validation_error", "error": str(e)})
                )
                last_validation = None

            # If validation passed, accept result
            if last_validation and last_validation.validation_passed:
                logger.info(
                    json.dumps(
                        {
                            "event": "synthesis_validated",
                            "attempt": attempt,
                            "valid_citations": last_validation.valid_citations,
                            "invalid_citations": last_validation.invalid_citations,
                        }
                    )
                )
                break

            # Otherwise, prepare targeted feedback and retry if attempts remain
            if attempt < max_attempts:
                invalids = last_validation.invalid_citations if last_validation else []
                feedback_text = (
                    "The previous synthesis included invalid or hallucinated citations: "
                    f"{invalids}.\nPlease regenerate the answer using ONLY the provided passages and ensure all inline citation numbers [N] refer to existing passages (1..{len(passages)})."
                )

                # Attach feedback via the contradictions field so the LLM sees the instruction
                feedback_contra = [
                    {
                        "topic": "Citation validation feedback",
                        "claim_1": {"paper_id": "meta_feedback", "text": feedback_text},
                        "claim_2": {
                            "paper_id": "meta_feedback",
                            "text": "Please correct citations and avoid hallucinations.",
                        },
                        "severity": "minor",
                    }
                ]

                logger.info(
                    json.dumps(
                        {
                            "event": "synthesis_retry_scheduled",
                            "attempt": attempt + 1,
                            "feedback": feedback_text,
                        }
                    )
                )

                # Set contradictions for the next attempt
                contradictions = feedback_contra
                continue

        # Update state with results
        new_state = state.model_copy()
        new_state.synthesized_answer = result.answer
        # Debug: log raw citations returned by agent for troubleshooting
        try:
            logger.info(
                f"synthesis result type: {type(result)}, citations attr type: {type(result.citations)}"
            )
            logger.info(json.dumps({"raw_citations": str(result.citations)}))
        except Exception:
            logger.info("raw_citations: <unserializable>")

        # Convert citations to plain dicts robustly: support pydantic models or plain dicts
        converted_citations = []
        for cit in result.citations or []:
            try:
                # pydantic model
                converted_citations.append(cit.model_dump())
            except Exception:
                try:
                    # plain dict-like
                    converted_citations.append(dict(cit))
                except Exception:
                    # Fallback to string representation
                    converted_citations.append({"text": str(cit)})

        new_state.citations = converted_citations
        # Fallback: if no structured citations were returned, try extracting inline [N] citations from the answer
        if not new_state.citations and isinstance(result.answer, str):
            import re

            matches = re.findall(r"\[(\d+)\]", result.answer)
            unique = []
            for m in matches:
                try:
                    idx = int(m)
                except Exception:
                    continue
                if idx < 1 or idx > len(passages):
                    continue
                if idx in unique:
                    continue
                unique.append(idx)
                passage = passages[idx - 1]
                new_state.citations.append(
                    {
                        "index": idx,
                        "paper_id": passage["paper_id"],
                        "text": passage["content"][:200]
                        + ("..." if len(passage["content"]) > 200 else ""),
                        "credibility_score": None,
                    }
                )
        new_state.consensus_estimate = result.consensus_estimate

        logger.info(
            "Synthesis node completed successfully",
            extra={
                "answer_length": len(result.answer),
                "citation_count": len(result.citations),
                "consensus": result.consensus_estimate,
            },
        )

        return new_state

    except Exception as e:
        logger.error(f"Synthesis node failed: {str(e)}")

        # Update state with error information
        new_state = state.model_copy()
        new_state.synthesized_answer = f"Error during synthesis: {str(e)}"
        new_state.citations = []
        new_state.consensus_estimate = None

        # Add to analysis errors for consistency
        if new_state.analysis_errors is None:
            new_state.analysis_errors = []
        new_state.analysis_errors.append(f"Synthesis failed: {str(e)}")

        return new_state


def build_workflow(request, mock_agents=None) -> StateGraph:
    """
    Build optimized LangGraph StateGraph workflow.

    Flow:
    1. optimize_query: Proactive query optimization with LLM
    2. search: OpenAlex-only search (100 papers, type filtering)
    3. deduplicate: Remove duplicate papers by DOI/title
    4. filter: BM25 + semantic filtering (100 → 25 papers)
    5. [fallback_refinement]: Backup refinement if search quality insufficient
    6. acquisition: Download PDFs for top 25 papers
    ... (continue with processing, analysis, synthesis)

    Args:
        request: FastAPI Request (for accessing app.state in agents)
        mock_agents: Optional dict of mock agents for testing

    Returns:
        Compiled StateGraph ready for execution
    """
    # Use mock agents if provided (for testing), otherwise use real agents
    query_optimizer = (
        mock_agents.get("query_optimizer") if mock_agents else QueryOptimizationAgent()
    )
    search_agent = mock_agents.get("search_agent") if mock_agents else SearchAgent()
    dedup_agent = (
        mock_agents.get("dedup_agent") if mock_agents else DeduplicationAgent()
    )
    filtering_agent = (
        mock_agents.get("filtering_agent") if mock_agents else FilteringAgent(request)
    )
    fallback_refiner = (
        mock_agents.get("fallback_refiner")
        if mock_agents
        else FallbackRefinementAgent()
    )
    acquisition_agent = (
        mock_agents.get("acquisition_agent") if mock_agents else AcquisitionAgent()
    )
    processing_agent = (
        mock_agents.get("processing_agent")
        if mock_agents
        else ProcessingAgent(request.app)
    )
    analysis_agent = (
        mock_agents.get("analysis_agent") if mock_agents else AnalysisAgent()
    )

    # Create StateGraph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("optimize_query", query_optimizer.optimize)
    workflow.add_node("search", search_agent.search)
    workflow.add_node("deduplicate", dedup_agent.deduplicate)
    workflow.add_node("filter", filtering_agent.filter_candidates)
    workflow.add_node("fallback_refinement", fallback_refiner.refine)
    workflow.add_node("acquisition", acquisition_agent)
    workflow.add_node("processing", processing_agent)
    workflow.add_node("analysis", analysis_agent.analyze)
    workflow.add_node("synthesis", synthesis_node)
    # ... add other nodes (verification)

    # Define edges (linear flow with conditional loops)
    workflow.set_entry_point("optimize_query")

    # Main flow
    workflow.add_edge("optimize_query", "search")

    # Conditional edge: Check if refinement needed after search
    workflow.add_conditional_edges(
        "search",
        should_refine_query,  # Routing function
        {
            "refine": "fallback_refinement",  # Trigger backup refinement
            "continue": "deduplicate",  # Continue to deduplication
        },
    )

    # Refinement loop (max 2 attempts)
    workflow.add_edge(
        "fallback_refinement", "search"
    )  # Retry search with refined query

    # Continue main flow
    workflow.add_edge("deduplicate", "filter")
    workflow.add_edge("filter", "acquisition")
    workflow.add_edge("acquisition", "processing")
    workflow.add_edge("processing", "analysis")
    workflow.add_edge("analysis", "synthesis")
    # ... add edges for remaining nodes

    # Compile workflow
    return workflow.compile()


# Test instantiation
if __name__ == "__main__":
    from models.state import State, Paper

    # Test State creation
    test_state = State(
        original_query="What are the cardiovascular risks of intermittent fasting?",
        papers_metadata=[
            Paper(
                paper_id="openalex:1",
                title="Paper 1",
                authors=["Author 1"],
                year=2023,
                journal="N/A",
                citation_count=0,
                doi="10.1000/1",
                abstract="Test abstract 1",
                pdf_url="https://example.com/paper1.pdf",
                type="article",
            ),
            Paper(
                paper_id="openalex:2",
                title="Paper 2",
                authors=["Author 2"],
                year=2023,
                journal="N/A",
                citation_count=0,
                doi="10.1000/2",
                abstract="Test abstract 2",
                pdf_url="https://example.com/paper2.pdf",
                type="review",
            ),
        ],
    )

    print(f"✅ State initialized: {test_state.request_id}")
    print(f"✅ Query: {test_state.query}")

    # Test workflow initialization
    class MockRequest:
        def __init__(self):
            self.app = type("MockApp", (), {})()

    workflow = build_workflow(MockRequest())
    print(f"✅ LangGraph workflow compiled successfully")

    # Example of running the workflow
    # result = workflow.invoke(test_state)
    # print(json.dumps(result.dict(), indent=2, default=str))
