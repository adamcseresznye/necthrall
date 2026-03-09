"""Query optimization agent for generating optimized search queries.

Transforms a user query into an optimized output containing:
- intent_type: Classification for ranking weight adjustment
- final_rephrase: Keyword-focused query for Semantic Scholar search
"""

from typing import Any, Dict, Optional

from loguru import logger

from config.prompts import QUERY_OPTIMIZATION_TEMPLATE
from utils.json_utils import parse_llm_json
from utils.llm_router import LLMRouter


class QueryOptimizationAgent:
    """Agent that optimizes user queries for Semantic Scholar search.

    Uses LLM to classify intent and rephrase queries.
    Handles LLM failures gracefully by falling back to original query.
    """

    def __init__(self) -> None:
        self.router = LLMRouter()

    async def optimize(self, query: str) -> Dict[str, Any]:
        """Optimize query for Semantic Scholar search.

        Args:
            query: The original user query string.

        Returns:
            Dict with keys 'intent_type' (str) and 'final_rephrase' (str).
            Falls back to original query on LLM failure.
        """
        logger.debug("QueryOptimizationAgent.optimize called with query: {}", query)

        prompt = self._build_prompt(query)
        response = await self._call_llm(prompt)

        if response is None:
            logger.warning("LLM call failed, using fallback")
            return self._fallback(query)

        parsed = self._parse_json_response(response)
        if parsed:
            logger.info(f"LLM Output structure: {parsed}")

        if parsed is None:
            logger.warning("JSON parsing failed, using fallback")
            return self._fallback(query)

        if not self._validate_response(parsed):
            logger.warning("LLM response missing required fields, using fallback")
            return self._fallback(query)

        logger.info(
            "Query optimization: intent_type='{}', final_rephrase='{}'",
            parsed["intent_type"],
            parsed["final_rephrase"],
        )
        return parsed

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM and handle timeouts/failures."""
        try:
            response = await self.router.generate(prompt, "optimization")
            logger.debug("LLM response received: {}", response[:200])
            return response
        except Exception as e:
            logger.exception("LLM call failed with exception: {}", e)
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM, handling markdown fences and literal newlines."""
        parsed = parse_llm_json(response)
        if parsed is None:
            logger.warning(
                "QueryOptimizationAgent raw LLM response (parse failed): {}", response
            )
        return parsed

    def _build_prompt(self, query: str) -> str:
        """Build the LLM prompt for query optimization."""
        return QUERY_OPTIMIZATION_TEMPLATE.format(query=query)

    def _validate_response(self, response: Dict) -> bool:
        """Validate that the LLM response contains required fields."""
        if not isinstance(response, dict):
            return False

        # Validate final_rephrase
        final_rephrase = response.get("final_rephrase")
        if not isinstance(final_rephrase, str) or not final_rephrase.strip():
            return False

        # Validate intent_type, default to "general" if invalid
        intent_type = response.get("intent_type", "general")
        if intent_type not in {"news", "foundational", "general"}:
            response["intent_type"] = "general"

        return True

    def _fallback(self, query: str) -> Dict[str, Any]:
        """Return fallback response using original query."""
        logger.debug("Using fallback response for query: {}", query)
        return {"intent_type": "general", "final_rephrase": query}


__all__ = ["QueryOptimizationAgent"]
