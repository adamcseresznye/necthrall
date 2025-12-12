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
import re


FEW_SHOT_EXAMPLE = """
### Executive Summary
**Intermittent fasting (IF) demonstrates consistent, modest benefits for cardiovascular health, primarily driven by weight loss and improved lipid profiles.**

While acute metabolic shifts—such as rapid TMAO reduction—are documented [2], long-term cardiovascular protection remains debated due to high attrition rates in trials (up to 38% [3]) and heterogeneity in fasting protocols [6][7]. Weight and BMI reduction through alternate-day fasting (ADF) achieves a 7% weight loss nadir at 6 months [3], with efficacy comparable to daily caloric restriction [12].


### Key Findings
* **Weight & BMI Reduction:** Alternate-day fasting (ADF) achieves a 7% weight loss nadir at 6 months [3], stabilizing at 4.5% below baseline at 12 months [3]. This efficacy is comparable to daily caloric restriction for overweight adults [12].
* **Biomarker Modulation:** A single 24-hour water-only fast reduced circulating TMAO from 27.1 to 14.3 ng/ml (p=0.019) [2].
* **Lipid Profile Improvement:** IF protocols generally lower total cholesterol [6] and LDL-C [7], though the magnitude of effect varies significantly by adherence levels.


### Synthesis & Implications
* **Mechanism of Action:** The benefits appear linked to both systemic weight reduction [3] and acute metabolic pauses that lower inflammatory markers like TMAO [2].
* **Protocol Viability:** While physiologically effective, the strictness of ADF leads to lower long-term adherence compared to less rigid restrictions [3], potentially limiting its utility as a public health intervention [12].


### Methodological Limits
* **Attrition Bias:** High dropout rates (38% in ADF arms [3]) likely inflate reported benefits by excluding non-adherent participants.
* **Transient vs. Chronic:** Key biomarkers like TMAO were often measured after acute fasting events [2], not reflecting long-term steady states.

---

❌ **INCORRECT EXAMPLE:**
"Fasting reduces TMAO and improves lipid profiles [2][3]."
*(Violates atomic citation rule—unclear which source supports which claim)*

✅ **CORRECT VERSION:**
"Fasting reduces TMAO [2] and improves lipid profiles [3]."
"""


CITATION_QA_TEMPLATE = (
    "You are a PhD-level research assistant. Your goal is to write a highly dense, rigorous synthesis. "
    "**Longer is NOT better.** Quality is defined by information density and thematic organization.\n"
    "---------------------\n"
    "### CONTEXT CHUNKS (Use ONLY these for citations):\n"
    "{context_str}\n"
    "---------------------\n\n"
    "### INSTRUCTIONS (Follow Strictly):\n"
    "1. **THEMATIC SYNTHESIS**:\n"
    "   - **DO NOT** organize bullets by study (e.g., avoid 'Study [1] said...').\n"
    "   - **DO** organize by **concepts, mechanisms, or outcomes**.\n"
    "   - Cite multiple sources within a single bullet if they support the same theme.\n\n"
    "2. **REQUIRED SECTIONS (Use ### Markdown Headers)**:\n"
    "   - **### Executive Summary**: \n"
    "       * Start with a **Bold 'Bottom Line' Sentence** summarizing the core conclusion.\n"
    "       * Add a line break.\n"
    "       * Follow with a concise, high-density paragraph (max 150 words).\n"
    "   - **### Key Findings**: Thematic bullet points with hard data.\n"
    "   - **### Synthesis & Implications**: Explain the relationship (e.g., trade-offs, complementarity). Replaces simple comparisons.\n"
    "   - **### Methodological Limits**: Brief critique (bullet points).\n\n"
    "3. **FORMATTING RULES**:\n"
    "   - **NO MARKDOWN TABLES**: Use nested bullet lists only.\n"
    "   - **BOLD LEAD-INS**: Start every bullet point with a **Bold Category:**.\n"
    "   - **ADAPTIVE COMPARISON**: \n"
    "       * **IF** the query compares entities (2 or more), treat every bullet as a 'table row'.\n"
    "       * **Rule:** Strictly repeat the 'vs.' pattern for every entity.\n"
    "       * **Format:** '**Category:** Entity A (Detail) **vs.** Entity B (Detail).'\n"
    "   - **NO IMAGES**: Do not generate image tags.\n\n"
    "4. **QUANTITATIVE DATA & CITATION PRECISION (CRITICAL)**: \n"
    "   - **NO OUTSIDE KNOWLEDGE**: You are strictly limited to the provided context chunks above. Do not use external knowledge.\n"
    "   - **NO HALLUCINATIONS**: If the text says 'expensive', **DO NOT** invent a number like '$500'. Write 'high cost' instead.\n"
    "   - **Specifics**: Use p-values, N=, and % changes **ONLY** if explicitly stated in the provided text.\n"
    "   - **ATOMIC CITATION RULE (MANDATORY)**: \n"
    "       * Each statistic or claim gets exactly ONE citation [N] immediately after it.\n"
    "       * **VERIFICATION STEP**: For every [N], ask yourself: 'Does passage N contain this exact number/claim?'\n"
    "       * **Correct:** 'Method A yield is 80% [1] vs. Method B is 40% [2].'\n"
    "       * **Incorrect:** 'Method A yield is 80% vs. Method B is 40% [1][2].'\n"
    "       * **Incorrect:** 'Method A and B yields are 80% and 40% respectively [1].' (Unless [1] contains BOTH numbers).\n"
    "   - **FORBIDDEN PATTERN**: Never group more than 2 citations together (e.g., `[1][2][3]` is prohibited).\n"
    "       * **Requirement:** If multiple sources support a statement, split the statement so each source supports a specific part.\n"
    "   - **CROSS-CHECK**: Before finalizing, re-read each citation and confirm the passage actually contains that specific fact.\n\n"
    "5. **CITATION VALIDATION (MANDATORY PRE-CHECK)**:\n"
    "   - Before writing each sentence with a factual claim, identify which source [N] supports it.\n"
    "   - If no source contains the information, write 'Evidence unavailable' or omit the claim entirely.\n"
    "   - **NEVER** cite [N] unless you can mentally quote the exact text from passage N that supports your claim.\n"
    "   - **PLANNING STEP**: For each key finding, create a mental note:\n"
    "       * 'Claim: [X]. Supporting passage: [N]. Key quote from passage: \"...\"'\n"
    "   - Then write the synthesis using only these validated claim-passage pairs.\n\n"
    "6. **START IMMEDIATELY**: \n"
    "   - Start your response directly with the header '### Executive Summary'.\n"
    "   - Do not use introductory labels like 'Scientific Answer:' or 'Here is the synthesis'.\n\n"
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
        temperature: Temperature for synthesis (default 0.15 for consistency).
    """

    INSUFFICIENT_CONTEXT_MESSAGE = "I cannot answer this based on the provided sources."

    def __init__(self, temperature: float = 0.15) -> None:
        """Initialize the synthesis agent.

        Args:
            temperature: LLM temperature for generation (0.15 = focused but not rigid).
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
            cleaned_response = self._clean_citations(response)
            logger.info(
                f"Synthesis completed successfully: response_length={len(cleaned_response)}"
            )
            return cleaned_response.strip()
        except Exception as e:
            logger.exception(f"Synthesis failed with error: {e}")
            # Re-raise to let caller decide how to handle
            raise

    def _clean_citations(self, text: str) -> str:
        """Clean OpenAI-style citation artifacts."""
        # 1. Fix brackets
        text = text.replace("【", "[").replace("】", "]")

        # 2. Remove '†' and everything after it (like 'source') until the closing bracket
        # This turns '[1†source]' into '[1]'
        return re.sub(r"†[^\]]*", "", text)

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
