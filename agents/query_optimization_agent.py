"""Query optimization agent for generating focused and variant queries.

Transforms a single user query into dual optimized outputs: a focused final_rephrase
for passage-level semantic retrieval, and three Semantic Scholar search variants
(primary, broad, alternative) for paper-level discovery.
"""

import json
import ast
from typing import Dict, Optional
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

    async def generate_dual_queries(self, query: str) -> Dict[str, str]:
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

        logger.info(
            "Query optimization successful: final_rephrase='{}', primary='{}', broad='{}', alternative='{}'",
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

        Your task: Extract the core research question from the user's input and generate four optimized variants.

        User input: "{query}"

        **Step 1: Extract Core Question**
        Identify the central research question. Ignore narrative details.

        **Step 2: Generate Four Query Variants**
        Create four distinct search queries. 
        CRITICAL: Semantic Scholar is a strict keyword search engine. 
        - DO NOT use boolean operators (AND, OR, +, -). 
        - DO NOT use full sentences for the 'primary', 'broad', or 'alternative' queries.
        - KEEP QUERIES SHORT (3-6 keywords max). Long queries return 0 results.

        1. **final_rephrase**: Refined natural language query for semantic retrieval (Vector Search)
        - Format: A clear, grammatically correct full sentence.
        - Example: "What are the molecular mechanisms linking intermittent fasting to cardiovascular risk?"

        2. **primary**: Precise keyword query
        - Format: 3-6 specific keywords.
        - Purpose: High precision matches.
        - Example: "intermittent fasting cardiovascular risk"

        3. **broad**: Expanded query
        - Format: 3-5 broad keywords or synonyms.
        - Purpose: High recall / category search.
        - Example: "time restricted eating metabolic health"

        4. **alternative**: Critical/Comparative query
        - Format: 3-6 keywords focusing on limitations or debates.
        - Purpose: Finding conflicting evidence or specific metrics.
        - Example: "fasting caloric restriction comparison safety"

        **IMPORTANT FORMATTING RULES:**
        - Return ONLY valid JSON.
        - Use DOUBLE QUOTES for all keys and string values.
        - No markdown.

        {{
        "final_rephrase": "...",
        "primary": "...",
        "broad": "...",
        "alternative": "..."
        }}"""

    def _validate_response(self, response: Dict) -> bool:
        """Validate that the LLM response contains all required fields."""
        required_keys = {"final_rephrase", "primary", "broad", "alternative"}
        return (
            isinstance(response, dict)
            and all(key in response for key in required_keys)
            and all(isinstance(response[key], str) for key in required_keys)
        )

    def _fallback(self, query: str) -> Dict[str, str]:
        """Return fallback response using original query for all fields."""
        logger.debug("Using fallback response for query: {}", query)
        return {
            "final_rephrase": query,
            "primary": query,
            "broad": query,
            "alternative": query,
        }


__all__ = ["QueryOptimizationAgent"]
