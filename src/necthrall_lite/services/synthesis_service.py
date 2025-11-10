import asyncio
import json
import time
from typing import List, Dict, Optional
from loguru import logger
import os


from utils.llm_client import get_safe_model_name

from ..api.schemas import SynthesisOutput, Citation, CitationValidation


class SynthesisAgent:
    """
    Agent for synthesizing evidence-backed answers from scientific paper passages.

    Uses LangChain's ChatGoogleGenerativeAI with temperature 0.3 for controlled creativity
    while maintaining strict factual grounding. Every claim is supported by inline citations.

    Features:
    - Generates 300-500 word answers with inline [N] citations
    - Handles contradictory evidence by presenting both sides
    - Integrates credibility scores to prioritize high-quality sources
    - Formats consensus estimates based on source agreement
    - Refuses to answer if evidence is insufficient

    Usage example:
        agent = SynthesisAgent()
        result = await agent.synthesize(
            query="What are the effects of intermittent fasting?",
            passages=[{"content": "...", "paper_id": "p1", ...}],
            credibility_scores=[{"paper_id": "p1", "score": 85, ...}],
            contradictions=[]
        )
    """

    def __init__(self, temperature: float = 0.3, max_tokens: int = 2500):
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLMs
        self.primary_llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL_PRIMARY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.fallback_llm = ChatGroq(
            model=os.getenv("LLM_MODEL_FALLBACK"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _invoke_llm_async(self, llm, messages, provider_name) -> Dict:
        """Invoke LLM asynchronously with logging and error handling."""
        try:
            logger.info(
                json.dumps({"event": "llm_call_start", "provider": provider_name})
            )
            start_time = time.time()
            response = await llm.ainvoke(messages)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(
                json.dumps(
                    {
                        "event": "llm_call_success",
                        "provider": provider_name,
                        "execution_time": execution_time,
                    }
                )
            )
            model_id = get_safe_model_name(llm)
            return {
                "content": response.content,
                "model_used": f"{provider_name}/{model_id}",
                "execution_time": execution_time,
            }
        except Exception as e:
            logger.warning(
                json.dumps(
                    {
                        "event": "llm_call_failure",
                        "provider": provider_name,
                        "error": str(e),
                    }
                )
            )
            raise e

    def _build_synthesis_prompt(
        self,
        query: str,
        passages: List[Dict],
        credibility_scores: List[Dict],
        contradictions: Optional[List[Dict]] = None,
    ) -> List:
        """Build the synthesis prompt with passages and instructions."""
        # Create credibility lookup
        cred_lookup = {cs["paper_id"]: cs for cs in credibility_scores}

        # Format passages with credibility
        formatted_passages = []
        for i, passage in enumerate(passages, 1):
            cred = cred_lookup.get(passage["paper_id"], {})
            score = cred.get("score", 0)
            tier = cred.get("tier", "low")
            formatted_passages.append(
                f"Passage {i} (Paper ID: {passage['paper_id']}, Credibility: {score}/100, Tier: {tier}):\n"
                f"Section: {passage.get('section', 'unknown')}\n"
                f"Content: {passage['content']}\n"
            )

        passages_text = "\n".join(formatted_passages)

        # Format contradictions if any
        contradictions_text = ""
        if contradictions:
            contra_list = []
            for contra in contradictions:
                contra_list.append(
                    f"- Topic: {contra['topic']}\n"
                    f"  Claim 1 (Paper {contra['claim_1']['paper_id']}): {contra['claim_1']['text']}\n"
                    f"  Claim 2 (Paper {contra['claim_2']['paper_id']}): {contra['claim_2']['text']}\n"
                    f"  Severity: {contra['severity']}\n"
                )
            contradictions_text = "Detected Contradictions:\n" + "\n".join(contra_list)

        system_prompt = """You are a scientific research assistant synthesizing evidence-backed answers.

TASK: Generate a 300-500 word answer to the query using ONLY the provided passages.
- Use inline citations [N] for EVERY claim, where N corresponds to the passage number.
- Prioritize high-credibility sources (higher scores, 'high'/'medium' tier).
- Present contradictory evidence objectively, showing both sides.
- Calculate consensus: "High consensus" (>80% agreement), "Moderate consensus" (50-80%), "Low consensus" (<50%), "Conflicting evidence".
- If evidence is insufficient or contradictory without resolution, refuse to answer definitively.

FORMAT:
- Answer in 300-500 words
- Use [N] citations inline
- End with consensus estimate
- Be factual, no hallucinations

OUTPUT FORMAT:
Answer text with [N] citations.

Consensus: [estimate]"""

        user_prompt = f"""Query: {query}

Passages:
{passages_text}

{contradictions_text}

Please synthesize an answer following the instructions."""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    def _parse_llm_response(
        self, response_content: str, passages: List[Dict]
    ) -> SynthesisOutput:
        """Parse LLM response into SynthesisOutput format."""
        # Extract answer and consensus
        lines = response_content.strip().split("\n")
        answer_lines = []
        consensus = None

        for line in lines:
            if line.lower().startswith("consensus:"):
                consensus = line.split(":", 1)[1].strip()
            else:
                answer_lines.append(line)

        answer = "\n".join(answer_lines).strip()

        # Extract citations from answer
        citations = []
        import re

        citation_matches = re.findall(r"\[(\d+)\]", answer)

        for match in citation_matches:
            idx = int(match)
            if 1 <= idx <= len(passages):
                passage = passages[idx - 1]
                citations.append(
                    Citation(
                        index=idx,
                        paper_id=passage["paper_id"],
                        text=(
                            passage["content"][:200] + "..."
                            if len(passage["content"]) > 200
                            else passage["content"]
                        ),
                        credibility_score=None,  # Will be filled later if needed
                    )
                )

        # Remove duplicates
        seen = set()
        unique_citations = []
        for cit in citations:
            if cit.index not in seen:
                seen.add(cit.index)
                unique_citations.append(cit)

        return SynthesisOutput(
            answer=answer, citations=unique_citations, consensus_estimate=consensus
        )

    def _validate_input(
        self,
        query: str,
        passages: List[Dict],
        credibility_scores: List[Dict],
        contradictions: Optional[List[Dict]] = None,
    ) -> None:
        """Validate input parameters."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not passages:
            raise ValueError("At least one passage required")

        if len(passages) > 10:
            raise ValueError("Maximum 10 passages supported")

        # Check passages have required fields
        for i, p in enumerate(passages):
            if not isinstance(p, dict):
                raise ValueError(f"Passage {i} must be a dictionary")
            if "content" not in p or "paper_id" not in p:
                raise ValueError(
                    f"Passage {i} missing required fields: content, paper_id"
                )

        # Check credibility scores
        if not credibility_scores:
            logger.warning("No credibility scores provided")

    @staticmethod
    def validate_citations(
        synthesized_answer: str, passages: List[Dict]
    ) -> CitationValidation:
        """
        Validate citations in a synthesized answer against available passages.

        Extracts all [N] citation patterns from the answer text, validates that each
        citation number N is within the valid range (1 to len(passages)), and generates
        detailed error reports for invalid citations.

        Args:
            synthesized_answer: The answer text containing [N] citation patterns
            passages: List of passage dictionaries that citations should reference

        Returns:
            CitationValidation object with validation results and error details

        Raises:
            ValueError: If passages list is None or empty

        Examples:
            >>> passages = [{"content": "test", "paper_id": "p1"}, {"content": "test2", "paper_id": "p2"}]
            >>> result = SynthesisAgent.validate_citations("Answer with [1] and [2]", passages)
            >>> result.validation_passed
            True

            >>> result = SynthesisAgent.validate_citations("Answer with [3]", passages)
            >>> result.validation_passed
            False
        """
        import re

        # Validate input
        if passages is None:
            raise ValueError("Passages list cannot be None")
        if not passages:
            raise ValueError("Passages list cannot be empty")

        num_passages = len(passages)

        # Compile regex pattern for performance
        citation_pattern = re.compile(r"\[\s*(\d+)\s*\]")

        # Extract all citation matches
        matches = citation_pattern.findall(synthesized_answer)

        # Convert to integers and get unique citations
        citation_numbers = set()
        for match in matches:
            try:
                citation_numbers.add(int(match))
            except ValueError:
                logger.warning(f"Invalid citation format: [{match}] - not a number")
                continue

        total_citations = len(citation_numbers)

        # Validate each citation
        valid_citations = 0
        invalid_citations = []
        error_details = []

        for citation_num in sorted(citation_numbers):
            if 1 <= citation_num <= num_passages:
                valid_citations += 1
            else:
                invalid_citations.append(citation_num)
                error_details.append(
                    f"Citation [{citation_num}] references non-existent passage "
                    f"(valid range: 1-{num_passages})"
                )

        validation_passed = len(invalid_citations) == 0

        # Log validation results
        logger.info(
            json.dumps(
                {
                    "event": "citation_validation",
                    "total_citations": total_citations,
                    "valid_citations": valid_citations,
                    "invalid_citations": len(invalid_citations),
                    "validation_passed": validation_passed,
                }
            )
        )

        return CitationValidation(
            total_citations=total_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            validation_passed=validation_passed,
            error_details=error_details,
        )

    async def synthesize(
        self,
        query: str,
        passages: List[Dict],
        credibility_scores: List[Dict],
        contradictions: Optional[List[Dict]] = None,
    ) -> SynthesisOutput:
        """
        Synthesize an evidence-backed answer from passages.

        Args:
            query: User's scientific question
            passages: Retrieved text passages with metadata
            credibility_scores: Paper credibility assessments
            contradictions: Optional detected contradictory claims

        Returns:
            SynthesisOutput with answer, citations, and consensus

        Raises:
            ValueError: If input validation fails
            Exception: If both LLMs fail
        """
        logger.info(
            "Starting synthesis",
            extra={"query_length": len(query), "passage_count": len(passages)},
        )

        # Validate input
        self._validate_input(query, passages, credibility_scores, contradictions)

        # Build prompt
        messages = self._build_synthesis_prompt(
            query, passages, credibility_scores, contradictions
        )

        # Try primary LLM
        try:
            response = await self._invoke_llm_async(
                self.primary_llm, messages, "gemini"
            )
        except Exception as e:
            logger.info("Primary LLM failed, trying fallback", extra={"error": str(e)})
            try:
                response = await self._invoke_llm_async(
                    self.fallback_llm, messages, "groq"
                )
            except Exception as e2:
                logger.error(
                    "Both LLMs failed",
                    extra={"primary_error": str(e), "fallback_error": str(e2)},
                )
                raise e2

        # Parse response
        result = self._parse_llm_response(response["content"], passages)

        logger.info(
            "Synthesis completed",
            extra={
                "answer_length": len(result.answer),
                "citation_count": len(result.citations),
            },
        )

        return result
