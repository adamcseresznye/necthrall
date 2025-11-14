"""Query optimization agent for generating focused and variant queries.

Transforms a single user query into dual optimized outputs: a focused final_rephrase
for passage-level semantic retrieval, and three Semantic Scholar search variants
(primary, broad, alternative) for paper-level discovery.

Usage example:
    agent = QueryOptimizationAgent()
    optimized = await agent.generate_dual_queries("intermittent fasting cardiovascular risks")
    # Returns: {
    #     "final_rephrase": "cardiovascular risks of intermittent fasting",
    #     "primary": "intermittent fasting cardiovascular risks",
    #     "broad": "fasting protocols health outcomes cardiovascular",
    #     "alternative": "time-restricted eating cardiac complications"
    # }
"""

import json
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
        """Parse JSON response from LLM."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # The LLM sometimes returns surrounding commentary or extra text
            # around the JSON blob. Try to extract the first top-level JSON
            # object by scanning for a balanced brace block and parse that.
            try:
                json_block = self._extract_json_block(response)
                if json_block:
                    parsed = json.loads(json_block)
                    return parsed
            except json.JSONDecodeError as e2:
                logger.warning("Extracted JSON block parsing failed: {}", e2)

            logger.exception(
                "JSON parsing failed: response not valid JSON or no parsable JSON block found"
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
        If the user provided a long explanation or story, identify the central research question. Ignore narrative details, background context, or personal anecdotes. Focus ONLY on what scientific information they're seeking.

        **Step 2: Generate Four Query Variants**
        Create four distinct search queries optimized for different purposes:

        1. **final_rephrase**: Focused technical query for passage-level semantic search
        - Purpose: Finding specific relevant passages within papers
        - Format: Concise phrase (5-10 words) with precise scientific terminology
        - Remove ambiguity, narrative elements, and filler words
        - Example: "intermittent fasting cardiovascular disease risk mechanisms"

        2. **primary**: Precise keyword query for Semantic Scholar paper search
        - Purpose: Exact matching in paper titles and abstracts
        - Format: 3-8 keywords capturing the core concept
        - Use specific technical terms that appear in academic papers
        - Optimized for Semantic Scholar's ranking algorithm
        - Example: "intermittent fasting cardiovascular risks"

        3. **broad**: Expanded query for comprehensive Semantic Scholar coverage
        - Purpose: Capturing related research and broader context
        - Format: 5-12 keywords including synonyms and related concepts
        - Include alternative terminology, related methods, and broader categories
        - Optimized for Semantic Scholar's keyword matching
        - Example: "fasting protocols cardiovascular health metabolic effects time-restricted eating"

        4. **alternative**: Creative reframing for Semantic Scholar discovery
        - Purpose: Finding adjacent research areas and alternative perspectives
        - Format: 4-10 keywords using different scientific framing
        - Explore different angles, mechanisms, or related phenomena
        - Optimized for Semantic Scholar's semantic understanding
        - Example: "caloric restriction cardiac function autophagy metabolic adaptation"

        **Semantic Scholar Query Best Practices:**
        - Use quotes for exact phrases: "red blood cell"
        - Use + for required terms: +cardiovascular
        - Use - to exclude terms: -animal (for human studies only)
        - Keep queries between 3-12 keywords for optimal results
        - Match terminology commonly found in paper titles/abstracts

        Return ONLY valid JSON with no additional text:

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
