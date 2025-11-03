from langgraph.graph import StateGraph, END
from models.state import State
from agents.search import SearchAgent
from agents.acquisition import AcquisitionAgent
from agents.filtering_agent import FilteringAgent
from agents.processing_agent import ProcessingAgent
from agents.query_optimization_agent import QueryOptimizationAgent
from agents.deduplication_agent import DeduplicationAgent
from agents.fallback_refinement_agent import FallbackRefinementAgent
from utils.logging_decorator import log_state_transition
from typing import Dict, Any
from loguru import logger
import json
import asyncio


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

    # Create StateGraph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("optimize_query", query_optimizer.optimize)
    workflow.add_node("search", search_agent.search)
    workflow.add_node("deduplicate", dedup_agent.deduplicate)
    workflow.add_node("filter", filtering_agent.filter_candidates)
    workflow.add_node("fallback_refinement", fallback_refiner.refine)
    workflow.add_node("acquisition", acquisition_agent)
    # ... add other nodes (processing, analysis, synthesis, verification)

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
