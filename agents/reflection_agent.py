"""Reflection agent for evaluating research completeness against a brief.

Given the persistent research_brief from PlanningAgent, the current answer
from SynthesisAgent, and all queries searched so far, determines whether
the research loop should continue and what gap to target next.
"""

from typing import List, Optional, TypedDict

from loguru import logger

from config.prompts import REFLECTION_TEMPLATE
from utils.json_utils import parse_llm_json
from utils.llm_router import LLMRouter


class ReflectionResult(TypedDict):
    is_complete: bool
    gap_description: str
    next_query: Optional[str]


class ReflectionAgent:
    """Agent that evaluates whether the current answer satisfies the research brief.

    Uses the research_brief as the anchor, not the previous answer.
    On any LLM or parsing failure, returns is_complete=True to avoid
    infinite loops (safe default).
    """

    def __init__(self) -> None:
        self.router = LLMRouter()

    async def evaluate(
        self,
        query: str,
        research_brief: str,
        current_answer: str,
        searched_queries: List[str],
    ) -> ReflectionResult:
        """Evaluate whether the current answer satisfies the research brief.

        Args:
            query: The original user research question.
            research_brief: Completeness definition from PlanningAgent (constant).
            current_answer: Answer produced by SynthesisAgent this round.
            searched_queries: All queries run so far in this research session.

        Returns:
            ReflectionResult with is_complete, gap_description, and next_query.
            On any failure, returns is_complete=True (safe default).
        """
        logger.debug(
            "ReflectionAgent.evaluate called — round {}, searched {} queries",
            len(searched_queries),
            len(searched_queries),
        )

        searched_str = "\n".join(f"- {q}" for q in searched_queries) or "None"
        prompt = REFLECTION_TEMPLATE.format(
            research_brief=research_brief,
            current_answer=current_answer,
            searched_queries=searched_str,
            query=query,
        )
        response = await self._call_llm(prompt)

        if response is None:
            logger.warning("LLM call failed in ReflectionAgent, using safe fallback")
            return self._fallback()

        parsed = self._parse_json_response(response)

        if parsed is None or not self._validate_response(parsed):
            logger.warning(
                "ReflectionAgent JSON parse/validation failed, using safe fallback"
            )
            return self._fallback()

        # Deduplicate: if next_query duplicates a searched query, force complete
        next_q = parsed.get("next_query")
        if next_q and next_q.strip().lower() in {
            q.strip().lower() for q in searched_queries
        }:
            logger.warning(
                "ReflectionAgent next_query '{}' duplicates a prior query — forcing is_complete=True",
                next_q,
            )
            return self._fallback()

        logger.info(
            "ReflectionAgent: is_complete={}, gap='{}'",
            parsed["is_complete"],
            parsed["gap_description"][:80] if parsed["gap_description"] else "",
        )
        return ReflectionResult(
            is_complete=bool(parsed["is_complete"]),
            gap_description=str(parsed.get("gap_description", "")),
            next_query=parsed.get("next_query"),
        )

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM router and surface any failures as None."""
        try:
            response = await self.router.generate(prompt, "optimization")
            logger.debug("LLM response received: {}", response[:200])
            return response
        except Exception as e:
            logger.exception("LLM call failed in ReflectionAgent: {}", e)
            return None

    def _parse_json_response(self, response: str) -> Optional[dict]:  # type: ignore[type-arg]
        """Parse JSON from LLM response, handling markdown fences and literal newlines."""
        parsed = parse_llm_json(response)
        if parsed is None:
            logger.warning("ReflectionAgent raw LLM response (parse failed): {}", response)
        return parsed

    def _validate_response(self, parsed: dict) -> bool:  # type: ignore[type-arg]
        """Validate required keys and types."""
        if "is_complete" not in parsed or not isinstance(parsed["is_complete"], bool):
            return False
        if "gap_description" not in parsed or not isinstance(
            parsed["gap_description"], str
        ):
            return False
        if "next_query" not in parsed:
            return False
        next_q = parsed["next_query"]
        if next_q is not None and not isinstance(next_q, str):
            return False
        # next_query must be None when is_complete is True
        if parsed["is_complete"] and next_q is not None:
            return False
        return True

    def _fallback(self) -> ReflectionResult:
        """Safe default — stops the loop on any failure."""
        return ReflectionResult(
            is_complete=True,
            gap_description="",
            next_query=None,
        )


__all__ = ["ReflectionAgent", "ReflectionResult"]
