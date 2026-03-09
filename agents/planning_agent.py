"""Planning agent for generating a research brief before literature search.

Produces three outputs from a single user query:
- research_brief: natural-language description of what 'done' looks like
- initial_query:  single broad Semantic Scholar keyword query
- final_rephrase: cleaned query for passage retrieval
"""

from typing import Optional, TypedDict

from loguru import logger

from config.prompts import PLANNING_TEMPLATE
from utils.json_utils import parse_llm_json
from utils.llm_router import LLMRouter


class PlanResult(TypedDict):
    research_brief: str
    initial_query: str
    final_rephrase: str


class PlanningAgent:
    """Agent that creates a research brief for a user query.

    Calls an LLM to define completeness criteria, an initial broad search query,
    and a clean rephrasing for semantic passage retrieval.
    Falls back to the original query on any LLM or parsing failure.
    """

    def __init__(self) -> None:
        self.router = LLMRouter()

    async def plan(self, query: str) -> PlanResult:
        """Generate a research plan for the given query.

        Args:
            query: The original user research question.

        Returns:
            PlanResult with research_brief, initial_query, and final_rephrase.
            On LLM failure, all three values fall back to the original query.
        """
        logger.debug("PlanningAgent.plan called with query: {}", query)

        prompt = PLANNING_TEMPLATE.format(query=query)
        response = await self._call_llm(prompt)

        if response is None:
            logger.warning("LLM call failed in PlanningAgent, using fallback")
            return self._fallback(query)

        parsed = self._parse_json_response(response)

        if parsed is None or not self._validate_response(parsed):
            logger.warning("PlanningAgent JSON parse/validation failed, using fallback")
            return self._fallback(query)

        logger.info(
            "PlanningAgent produced brief ({} chars), initial_query='{}'",
            len(parsed["research_brief"]),
            parsed["initial_query"],
        )
        return parsed  # type: ignore[return-value]

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM router and surface any failures as None."""
        try:
            response = await self.router.generate(prompt, "optimization", max_tokens=1024)
            logger.debug("LLM response received: {}", response[:200])
            return response
        except Exception as e:
            logger.exception("LLM call failed in PlanningAgent: {}", e)
            return None

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown fences and literal newlines."""
        parsed = parse_llm_json(response)
        if parsed is None:
            logger.warning("PlanningAgent raw LLM response (parse failed): {}", response)
        return parsed

    def _validate_response(self, parsed: dict) -> bool:
        """Validate that all three required keys are present and non-empty strings."""
        required = {"research_brief", "initial_query", "final_rephrase"}
        if not required.issubset(parsed.keys()):
            return False
        return all(isinstance(parsed[k], str) and parsed[k].strip() for k in required)

    def _fallback(self, query: str) -> PlanResult:
        """Return the original query for all three fields on any failure."""
        return PlanResult(
            research_brief=query,
            initial_query=query,
            final_rephrase=query,
        )


__all__ = ["PlanningAgent", "PlanResult"]
