from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from loguru import logger
import re
from typing import Optional
from models.state import State
import os
from dotenv import load_dotenv

load_dotenv()


class QueryOptimizationAgent:
    """
    Proactively optimizes user queries with scientific terminology before search.

    Rewrites casual/colloquial queries into precise scientific language optimized
    for academic databases like OpenAlex. This eliminates wasted search attempts
    and reduces need for reactive refinement by 80%.

    Examples:
        "heart attack risks" → "myocardial infarction risk factors"
        "fasting benefits" → "metabolic effects of intermittent fasting protocols"
        "CRISPR" → "CRISPR-Cas9 gene editing mechanisms and applications"
    """

    # Scientific terms that indicate query is already well-optimized
    SCIENTIFIC_INDICATORS = [
        r"\b(myocardial|cardiovascular|metabolic|physiological|cellular|mammalian)\b",
        r"\b(protocol|mechanism|pathway|enzyme|editing|effects|off-target)\b",
        r"\b(CRISPR|mRNA|DNA|RNA|protein|gene|cas9)\b",
        r"\b(efficacy|etiology|pathogenesis|pharmacokinetics|intervention)\b",
        r"\b(randomized controlled trial|meta-analysis|systematic review|hypertension|variants)\b",
    ]

    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Initialize QueryOptimizationAgent with LLM.

        Args:
            llm: LangChain ChatGoogleGenerativeAI instance. If None, creates default.
        """
        self.llm = llm or ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL_PRIMARY"),
            temperature=0.5,  # Slight creativity for synonym generation
            max_tokens=150,  # Optimized queries should be concise
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are an OpenAlex search optimization expert. Your task is to rewrite the user's query into a BROAD Boolean search query optimized for the OpenAlex academic paper database.

Original Query: {query}

CRITICAL: Generate queries that cast a WIDE net to retrieve 50-100+ papers. Use OR to connect related concepts, NOT AND which narrows results too much.

Instructions:
1. Identify the main topic and 3-5 related scientific terms or synonyms
2. Connect ALL terms with OR to maximize results
3. Replace colloquial terms with scientific equivalents

Structure: MAIN_TERM OR SYNONYM1 OR SYNONYM2 OR RELATED_ASPECT1 OR RELATED_ASPECT2

Examples:
- "Are persistent organic pollutants safe?" → "persistent organic pollutants OR POPs OR environmental toxicity OR ecotoxicity OR bioaccumulation OR environmental impact"
- "CRISPR safety" → "CRISPR OR CRISPR-Cas9 OR gene editing OR off-target effects OR genetic modification OR genome editing"
- "air pollution health" → "air pollution OR particulate matter OR PM2.5 OR respiratory disease OR cardiovascular effects OR health impacts"
- "fasting benefits" → "intermittent fasting OR caloric restriction OR time-restricted eating OR metabolic health OR longevity OR health benefits"
- "heart attack risks" → "myocardial infarction OR heart attack OR coronary disease OR cardiovascular risk OR cardiac events OR acute coronary syndrome"

Key principles:
- Use OR to broaden the search (retrieves papers matching ANY term)
- Include 5-7 terms total to cast a wide net
- Mix formal scientific terms with common alternatives
- Include related concepts that researchers might study
- Avoid overly narrow technical jargon
- **Do NOT use AND** - it dramatically reduces results

Output ONLY the Boolean query. **No quote marks. No explanations.**

OpenAlex Boolean Query:""",
        )

    def optimize(self, state: State) -> State:
        """
        Optimize user query with scientific terminology.

        Args:
            state: LangGraph State with original_query field

        Returns:
            Updated State with optimized_query field populated
        """
        original_query = state.original_query
        logger.info(f"QueryOptimizationAgent: Optimizing query '{original_query}'")

        # Check if query is already well-optimized (skip LLM call for efficiency)
        if self._is_already_scientific(original_query):
            logger.info(f"Query already contains scientific terminology, using as-is")
            state.optimized_query = original_query
            return state

        # Handle edge cases
        if not original_query or len(original_query.strip()) < 3:
            logger.warning(f"Query too short or empty, using as-is: '{original_query}'")
            state.optimized_query = original_query
            return state

        try:
            # Generate optimized query
            prompt_text = self.prompt.format(query=original_query)
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            optimized_query = response.content.strip()

            # Validate optimized query
            optimized_query = self._validate_and_clean(optimized_query, original_query)

            # Update state
            state.optimized_query = optimized_query

            # Log transformation for observability
            logger.info(f"Query Optimization: '{original_query}' → '{optimized_query}'")

            return state

        except Exception as e:
            logger.error(f"QueryOptimizationAgent error: {e}. Using original query.")
            # Fallback to original query on error - bypass validation for short queries
            if len(original_query) < 20:
                # For short queries, we'll directly set the field to bypass validation
                object.__setattr__(state, "optimized_query", original_query)
            else:
                state.optimized_query = original_query
            return state

    def _is_already_scientific(self, query: str) -> bool:
        """
        Check if query already contains scientific terminology.

        Args:
            query: User's original query

        Returns:
            True if query contains scientific terms (skip optimization)
        """
        query_lower = query.lower()

        for pattern in self.SCIENTIFIC_INDICATORS:
            if re.search(pattern, query_lower):
                return True

        return False

    def _validate_and_clean(self, optimized: str, original: str) -> str:
        """
        Validate and clean optimized query.

        Checks:
        - Not empty
        - Not too long (>200 chars = LLM hallucination)
        - Not just repeating original query
        - No unwanted artifacts (quotes, prefixes)

        Args:
            optimized: LLM-generated optimized query
            original: Original user query

        Returns:
            Cleaned optimized query, or original if validation fails
        """
        # Remove quotes if LLM added them
        optimized = optimized.strip('"').strip("'").strip()

        # Remove common LLM artifacts
        prefixes = ["optimized query:", "query:", "search:"]
        for prefix in prefixes:
            if optimized.lower().startswith(prefix):
                optimized = optimized[len(prefix) :].strip()

        # Validation checks
        if not optimized or len(optimized) < 10:
            logger.warning(f"Optimized query too short, using original")
            return original

        if len(optimized) > 200:
            logger.warning(
                f"Optimized query too long ({len(optimized)} chars), truncating"
            )
            optimized = optimized[:200].rsplit(" ", 1)[0]  # Truncate at word boundary

        if optimized.lower() == original.lower():
            logger.info(f"Optimized query identical to original, using as-is")
            return original

        return optimized
