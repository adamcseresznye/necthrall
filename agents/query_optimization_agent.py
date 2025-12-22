"""Query optimization agent for generating focused and variant queries.

Transforms a single user query into dual optimized outputs: a focused final_rephrase
for passage-level semantic retrieval, and three Semantic Scholar search variants
(primary, broad, alternative) for paper-level discovery.
"""

import json
import ast
from typing import Dict, Optional, Any
from loguru import logger

from utils.llm_router import LLMRouter


class QueryOptimizationAgent:
    """Agent that optimizes user queries for hybrid search strategy.

    Uses LLM to generate four query variants:
    - final_rephrase: Focused query for passage retrieval
    - primary: Most specific Semantic Scholar query
    - broad: Wider coverage query with synonyms
    - alternative: Different terminology/framing

    Handles LLM failures gracefully by falling back to original query.
    """

    def __init__(self) -> None:
        self.router = LLMRouter()

    async def generate_dual_queries(self, query: str) -> Dict[str, Any]:
        """Generate dual optimized query outputs using LLM.

        Args:
            query: The original user query string.

        Returns:
            Dict with keys: final_rephrase, primary, broad, alternative.
            All values are strings. On LLM failure, all values equal the input query.
        """
        logger.debug(
            "QueryOptimizationAgent.generate_dual_queries called with query: {}", query
        )

        prompt = self._build_prompt(query)
        response = await self._call_llm(prompt)

        if response is None:
            logger.warning("LLM call failed, using fallback")
            return self._fallback(query)

        parsed = self._parse_json_response(response)
        if parsed is None:
            logger.warning("JSON parsing failed, using fallback")
            return self._fallback(query)

        if not self._validate_response(parsed):
            logger.warning("LLM response missing required fields, using fallback")
            return self._fallback(query)

        # Log based on strategy
        strategy = parsed.get("strategy", "expansion")
        if strategy == "decomposition":
            logger.info(
                "Query optimization (Decomposition): final_rephrase='{}', sub_queries={}",
                parsed["final_rephrase"],
                parsed["sub_queries"],
            )
        else:
            logger.info(
                "Query optimization (Expansion): final_rephrase='{}', primary='{}', broad='{}', alternative='{}'",
                parsed["final_rephrase"],
                parsed["primary"],
                parsed["broad"],
                parsed["alternative"],
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
        """Parse JSON response from LLM, handling both JSON and Python-dict formats."""
        # 1. Try strict JSON parsing first (fastest/safest)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 2. Extract the block (handles markdown fences like ```json ... ```)
        json_block = self._extract_json_block(response)
        if not json_block:
            logger.error("No JSON block found in response")
            return None

        # 3. Try strict JSON parsing on the extracted block
        try:
            return json.loads(json_block)
        except json.JSONDecodeError as e:
            # 4. Fallback: Try ast.literal_eval for Python-style dicts (single quotes)
            # This fixes "Expecting property name enclosed in double quotes"
            try:
                # ast.literal_eval safely evaluates a string containing a Python literal
                return ast.literal_eval(json_block)
            except (ValueError, SyntaxError):
                logger.warning(
                    "Parsing failed via both json.loads and ast.literal_eval. Error: {}",
                    e,
                )
                return None

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON object from text.

        This scans for the first '{' and then finds the matching '}' by
        tracking brace depth. Returns the substring (including braces) or
        None when no balanced block is found.
        """
        if not text or "{" not in text:
            return None

        start = text.find("{")
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _build_prompt(self, query: str) -> str:
        """Build the LLM prompt for query optimization."""
        return f"""You are a query optimization expert for scientific research using Semantic Scholar API.

        Your task: Analyze the user's query and choose the best strategy: 'expansion' or 'decomposition'.

        User input: "{query}"

        **Strategy A: Expansion (Default)**
        Use this for single-topic or straightforward queries.
        Generate three keyword-based variants for Semantic Scholar and one natural language rephrase.

        Output Format (JSON):
        {{
            "strategy": "expansion",
            "final_rephrase": "Clear natural language question for semantic search",
            "primary": "3-6 specific keywords",
            "broad": "3-5 broad keywords or synonyms",
            "alternative": "3-6 keywords focusing on limitations or debates"
        }}

        **Strategy B: Decomposition**
        Use this for complex, multi-part, or comparative queries that require breaking down.
        Generate a list of sub-queries to be executed independently.

        Output Format (JSON):
        {{
            "strategy": "decomposition",
            "final_rephrase": "The overarching question in clear natural language",
            "sub_queries": [
                "First sub-question keywords",
                "Second sub-question keywords",
                "..."
            ]
        }}

        **CRITICAL RULES:**
        - 'final_rephrase' is MANDATORY for BOTH strategies.
        - For 'primary', 'broad', 'alternative', and 'sub_queries': DO NOT use boolean operators (AND, OR). Keep them short (3-6 keywords).
        - Return ONLY valid JSON.
        """

    def _validate_response(self, response: Dict) -> bool:
        """Validate that the LLM response contains all required fields based on strategy."""
        if not isinstance(response, dict):
            return False

        strategy = response.get("strategy", "expansion")

        # Common mandatory field
        if "final_rephrase" not in response or not isinstance(
            response["final_rephrase"], str
        ):
            return False

        if strategy == "decomposition":
            return (
                "sub_queries" in response
                and isinstance(response["sub_queries"], list)
                and all(isinstance(q, str) for q in response["sub_queries"])
            )
        else:
            # Expansion strategy (default)
            required_keys = {"primary", "broad", "alternative"}
            return all(key in response for key in required_keys) and all(
                isinstance(response[key], str) for key in required_keys
            )

    def _fallback(self, query: str) -> Dict[str, Any]:
        """Return fallback response using original query for all fields."""
        logger.debug("Using fallback response for query: {}", query)
        return {
            "strategy": "expansion",
            "final_rephrase": query,
            "primary": query,
            "broad": query,
            "alternative": query,
        }


__all__ = ["QueryOptimizationAgent"]
