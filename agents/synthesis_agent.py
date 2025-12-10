"""Synthesis agent for generating cited answers from retrieved passages.

This agent takes a list of retrieved passages (as LlamaIndex NodeWithScore objects)
and generates a final answer with inline citations (e.g., [1], [2]) using the
configured LLM via LiteLLM routing.

Usage:
    from agents.synthesis_agent import SynthesisAgent
    from llama_index.core.schema import NodeWithScore, TextNode

    agent = SynthesisAgent()
    answer = await agent.synthesize(
        query="What are the cardiovascular risks of intermittent fasting?",
        nodes=[node1, node2, node3]
    )
    # Returns: "Intermittent fasting may affect heart rate variability [1] and..."

Architecture:
    - Uses LLMRouter for model selection with primary/fallback
    - Enforces strict [N] citation format to prevent hallucinations
    - Handles empty nodes gracefully with fallback message
"""

from typing import List
from loguru import logger

from llama_index.core.schema import NodeWithScore

from utils.llm_router import LLMRouter


FEW_SHOT_EXAMPLE = """
User Query: cardiovascular effects of intermittent fasting
Scientific Answer:
### Executive Summary
Across the available evidence, intermittent fasting (IF) is associated with modest but consistent improvements in cardiovascular risk factors. Key outcomes include reductions in body weight (approx. 4-7% [3]), modest declines in blood pressure, and acute lowering of TMAO (from 27.1 to 14.3 ng/ml [2]). However, methodological limitations, such as high attrition rates (38% in ADF trials [3]), temper definitive conclusions.

### Key Findings
* **Body Weight & BMI:** Alternate-day fasting (ADF) produced a 7% weight loss nadir at 6 months, stabilizing at 4.5% below baseline at 12 months [3].
* **TMAO (Biomarker):** A 24-hour water-only fast reduced circulating TMAO from 27.1 ng/ml to 14.3 ng/ml (p=0.019), indicating a rapid metabolic shift [2].
* **Lipid Profile:** IF is described to lower total cholesterol and LDL-C, though quantitative trial data varies by protocol [6][7].

### Detailed Comparison
* **Weight Loss**
    * **Study [3]:** Randomized trial (N=not specified) compared ADF vs. daily caloric restriction. Both groups achieved ~7% loss at 6 months.
    * **Study [12]:** Meta-analysis of RCTs (n=104 outcomes) confirmed moderate reductions in BMI for overweight adults.
* **Inflammation**
    * **Study [6]:** Notes reductions in systemic inflammation markers.
    * **Study [2]:** Re-feeding after 24h fast restored baseline TMAO levels, suggesting transient effects.

### Limitations & Quality of Evidence
* **Attrition:** The 12-month ADF trial had a 38% dropout rate, potentially biasing results toward adherent participants [3].
* **Short-Term Data:** Many metabolic changes (e.g., TMAO) were measured after a single 24h fast, limiting insight into long-term impact [2].
"""


CITATION_QA_TEMPLATE = (
    "You are a PhD-level research assistant. Your goal is to write a highly dense, rigorous synthesis. "
    "**Longer is NOT better.** Quality is defined by information density.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "### INSTRUCTIONS (Follow Strictly):\n"
    "1. **ADAPTIVE LENGTH STRATEGY (CRITICAL)**:\n"
    "   - **Step 1**: Evaluate the retrieved chunks. Are they rich or sparse?\n"
    "   - **Step 2**: If data is sparse, write a short, direct answer (e.g., 100 words). **Do not** fluff up the text to fill space.\n"
    "   - **Step 3**: If data is rich, synthesize it concisely. Merge related findings into single dense sentences.\n\n"
    "2. **REQUIRED SECTIONS**:\n"
    "   - **Executive Summary**: The distilled conclusion. Maximum 200 words, but shorter is preferred if possible.\n"
    "   - **Key Findings**: Bullet points of hard data only. If no hard numbers exist, skip this section or keep it to 1-2 points.\n"
    "   - **Detailed Comparison**: Only include if there are distinct protocols/groups to compare. Otherwise, merge into findings.\n"
    "   - **Methodological Limits**: Brief critique (bullet points).\n\n"
    "3. **QUANTITATIVE ENFORCEMENT**: \n"
    "   - Use specific numbers (p-values, N=, % change) whenever available.\n"
    "   - WARNING: Do not invent numbers. If data is missing, write 'quantities not reported'.\n"
    "   - Cite every claim immediately [x].\n\n"
    "4. **FORMATTING RULES**:\n"
    "   - **NO MARKDOWN TABLES**: Use nested bullet lists only.\n"
    "   - **NO IMAGES**: Do not generate image tags or markdown images.\n"
    "   - **NO PREAMBLE**: Start directly with '### Executive Summary'.\n\n"
    "5. **ACCESSIBILITY** (optional):\n"
    "   - Define technical terms on first use in parentheses, e.g., 'VO₂peak (maximal oxygen uptake)'.\n"
    "   - Explain acronyms: 'coronary artery disease (CAD)'.\n\n"
    "### REQUIRED OUTPUT FORMAT:\n"
    "---------------------\n"
    f"{FEW_SHOT_EXAMPLE}\n"
    "---------------------\n\n"
    "User Query: {query_str}\n"
)


class SynthesisAgent:
    """Agent that synthesizes answers from retrieved passages with citations.

    Uses LLMRouter to call the configured synthesis model (with fallback).
    Formats passages with indices and enforces citation requirements via
    a carefully crafted prompt template.

    Attributes:
        router: The LLMRouter instance for LLM calls.
        temperature: Temperature for synthesis (default 0.3 for consistency).
    """

    INSUFFICIENT_CONTEXT_MESSAGE = "I cannot answer this based on the provided sources."

    def __init__(self, temperature: float = 0.3) -> None:
        """Initialize the synthesis agent.

        Args:
            temperature: LLM temperature for generation (0.3 = focused but not rigid).
        """
        self.router = LLMRouter()
        self.temperature = temperature
        logger.info(f"SynthesisAgent initialized with temperature={temperature}")

    async def synthesize(self, query: str, nodes: List[NodeWithScore]) -> str:
        """Synthesize a cited answer from query and retrieved nodes.

        Args:
            query: The user's original question.
            nodes: List of NodeWithScore objects from retrieval (hybrid search results).

        Returns:
            A synthesized answer string containing inline [N] citations.
            Returns fallback message if nodes are empty or synthesis fails.

        Raises:
            Exception: Re-raises LLM errors after logging (allows caller to handle).
        """
        logger.debug(
            f"SynthesisAgent.synthesize called: query='{query[:50]}...' nodes_count={len(nodes)}"
        )

        # Handle empty nodes gracefully
        if not nodes:
            logger.warning(
                "No nodes provided for synthesis, returning fallback message"
            )
            return self.INSUFFICIENT_CONTEXT_MESSAGE

        # Build context string with passage indices
        context_str = self._format_context(nodes)

        # Build the full prompt
        prompt = CITATION_QA_TEMPLATE.format(
            context_str=context_str,
            query_str=query,
        )

        logger.debug(
            f"Synthesis prompt built: {len(prompt)} chars, {len(nodes)} passages"
        )

        try:
            # Call LLM via router (handles primary/fallback)
            response = await self.router.generate(prompt, model_type="synthesis")
            logger.info(
                f"Synthesis completed successfully: response_length={len(response)}"
            )
            return response.strip()
        except Exception as e:
            logger.exception(f"Synthesis failed with error: {e}")
            # Re-raise to let caller decide how to handle
            raise

    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        """Format nodes into a context string with passage indices.

        Each passage is formatted as:
            [N] <passage text>
            ---

        Args:
            nodes: List of NodeWithScore objects.

        Returns:
            Formatted context string for the prompt.
        """
        passages = []
        for idx, node_with_score in enumerate(nodes):
            # Extract text from the node
            text = self._extract_text(node_with_score)
            # Format with 1-based index for human readability
            passage_idx = idx + 1
            passages.append(f"[{passage_idx}] {text}")

        return "\n---\n".join(passages)

    def _extract_text(self, node_with_score: NodeWithScore) -> str:
        """Extract text content from a NodeWithScore object.

        Handles both direct text access and get_content() method.

        Args:
            node_with_score: A NodeWithScore from retrieval.

        Returns:
            The text content of the node.
        """
        node = node_with_score.node
        # Try get_content() first (standard LlamaIndex method)
        if hasattr(node, "get_content"):
            return node.get_content()
        # Fallback to text attribute
        if hasattr(node, "text"):
            return node.text
        # Last resort: string conversion
        logger.warning(f"Node has no text extraction method, using str(): {type(node)}")
        return str(node)


__all__ = ["SynthesisAgent", "CITATION_QA_TEMPLATE"]
