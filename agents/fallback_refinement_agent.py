from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import logging
from typing import Optional
from models.state import State
import os

logger = logging.getLogger(__name__)


class FallbackRefinementAgent:
    """
    Backup refinement agent triggered when proactive optimization isn't sufficient.

    Differences from old QueryRefinementAgent:
    - Role changed from PRIMARY to BACKUP refinement strategy
    - Prompt includes context about why proactive optimization failed
    - Uses search_quality feedback to guide refinement

    Only triggered in ~20% of queries where:
    - Proactive QueryOptimizationAgent produced weak results
    - SearchAgent returned <10 papers or avg_relevance <0.4
    """

    def __init__(self, llm=None):
        """Initialize FallbackRefinementAgent with LLM."""
        # If explicitly set to None (for testing), keep as None
        if llm is None:
            self.llm = None
        else:
            # Try to create LLM for non-testing environments
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=os.getenv("LLM_MODEL_PRIMARY", "gemini-2.0-flash-exp"),
                    temperature=0.6,  # Slightly higher than optimization for more creativity
                    max_tokens=150,
                )
            except Exception:
                # For testing environments without credentials
                self.llm = None

        self.prompt = PromptTemplate(
            input_variables=["original_query", "optimized_query", "search_summary"],
            template="""You are a scientific search expert performing BACKUP query refinement.

Context:
- Original user query: "{original_query}"
- Proactively optimized query: "{optimized_query}"
- Search result: {search_summary}

The proactive optimization wasn't sufficient. The search returned poor results, indicating the optimized query needs further refinement.

Your task: Further refine the optimized query to improve search results.

Refinement strategies:
1. Add more specific mechanisms, pathways, or methodologies:
   - "intermittent fasting" → "AMPK and mTOR pathway modulation during intermittent fasting"
   - "cancer treatment" → "immune checkpoint inhibitor mechanisms in oncology"

2. Include synonyms or alternative terminology:
   - "gene editing" → "CRISPR-Cas9 and TALEN-based genome editing"
   - "cognitive decline" → "Alzheimer disease and neurodegenerative disorders"

3. Narrow scope if query is too broad:
   - "diabetes" → "type 2 diabetes mellitus insulin resistance mechanisms"
   - "nutrition" → "macronutrient composition and metabolic health outcomes"

4. Add time constraints or population specificity if missing:
   - "fasting benefits" → "metabolic effects of intermittent fasting in adults"

Important:
- Don't repeat the exact same query
- Maintain scientific terminology (don't regress to casual language)
- Keep between 8-15 words
- Output ONLY the refined query, no explanations

Refined Query:""",
        )

    def refine(self, state: State) -> State:
        """
        Perform backup refinement when proactive optimization wasn't sufficient.

        Args:
            state: LangGraph State with search_quality indicating failure

        Returns:
            Updated State with refined optimized_query and incremented refinement_count
        """
        original = state.original_query
        optimized = state.optimized_query or original
        search_quality = state.search_quality or {}

        # Build search summary for context
        search_summary = (
            f"{search_quality.get('paper_count', 0)} papers found, "
            f"avg_relevance {search_quality.get('avg_relevance', 0):.2f}"
        )

        logger.info(
            f"FallbackRefinementAgent: Refining query (attempt {state.refinement_count + 1}/2). "
            f"Search failed: {search_quality.get('reason', 'Unknown')}"
        )

        try:
            # Handle testing case where LLM is not available
            if self.llm is None:
                logger.warning("LLM not available, using fallback refinement")
                refined_query = f"refined {optimized}"
            else:
                # Generate refined query
                prompt_text = self.prompt.format(
                    original_query=original,
                    optimized_query=optimized,
                    search_summary=search_summary,
                )
                response = self.llm.invoke([HumanMessage(content=prompt_text)])
                refined_query = response.content.strip().strip('"').strip("'")

            # Validation
            if not refined_query or len(refined_query) < 10:
                logger.warning("Refined query too short, using optimized query")
                refined_query = optimized

            if refined_query.lower() == optimized.lower():
                logger.warning("Refined query identical to optimized, using as-is")

            # Update state
            state.optimized_query = refined_query
            state.refinement_count += 1

            logger.info(
                f"FallbackRefinementAgent: '{optimized}' → '{refined_query}' "
                f"(refinement {state.refinement_count}/2)"
            )

            return state

        except Exception as e:
            logger.error(
                f"FallbackRefinementAgent error: {e}. Keeping optimized query."
            )
            state.refinement_count += 1  # Increment to prevent infinite loop
            return state
