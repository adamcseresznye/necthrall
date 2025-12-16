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
from config import RAG_RERANK_TOP_K

from utils.llm_router import LLMRouter
import re


FEW_SHOT_EXAMPLE = """
**Verdict: Yes, with consensus on mechanism but debate on magnitude.**
Rapamycin consistently extends lifespan in model organisms via mTOR inhibition [1], though sexual dimorphism in mice remains a significant variable [3].

### Evidence Synthesis
The extension of lifespan by rapamycin is robust and reproducible across diverse taxa, including yeast, nematodes, and mice [1]. The primary mechanism is the inhibition of the *mammalian target of rapamycin* (mTOR) pathway, which mimics caloric restriction and enhances autophagy [4]. In murine models, treatment initiated even in late life (600 days) significantly increases survival rates [2], suggesting the intervention is effective even after aging has commenced.

### Critical Nuances & Conflicts
* **Sexual Dimorphism –** Evidence suggests a stronger effect in females than males. One major study found a 14% extension in females versus only 9% in males at the same dosage [3], potentially due to differences in hepatic drug metabolism.
* **Dosage Toxicity –** While lifespan is extended, high doses are associated with testicular degeneration [5], indicating a narrow therapeutic window.
"""

CITATION_QA_TEMPLATE = (
    "You are a Senior Scientific Research Fellow briefing a Principal Investigator. "
    "Your goal is to distill a complex body of literature into a definitive, scientifically rigorous synthesis.\n"
    "---------------------\n"
    "### INSTRUCTIONS:\n"
    "1. **THE BLUF (Bottom Line Up Front):**\n"
    "   - Start immediately with a bold **Label**. Choose the best fit:\n"
    "       * *Binary:* **Verdict: Yes / No / Mixed.**\n"
    "       * *Definitional:* **Core Concept: [Phrase].**\n"
    "       * *Methodological:* **Standard Protocol: [Method].**\n"
    "       * *Open:* **Scientific Consensus: [Theme].**\n"
    "   - Follow with a high-level thesis sentence summarizing the answer.\n\n"
    "2. **THE EVIDENCE (Structured & Rigorous):**\n"
    "   - **Evidence Synthesis:** Synthesize the high-authority agreement. What is the established truth? (Cite support).\n"
    "   - **Critical Nuances:** Discuss conflicts, sexual dimorphism, in vivo vs in vitro discrepancies, or major limitations.\n\n"
    "3. **STYLE & CONSTRAINTS:**\n"
    "   - **Target Length:** ~250-350 words. Be dense but readable.\n"
    "   - **Bullet Style:** Start every bullet point with a **Bold Concept Label** followed by a dash. Mandatory.\n"
    "   - **Tone:** Professional. Use precise terminology.\n"
    "   - **Definitions:** Define ONLY non-standard acronyms on first use.\n\n"
    "4. **PROTOCOL FOR INSUFFICIENT DATA:**\n"
    "   - If the provided chunks do not contain the answer, do not hallucinate.\n"
    "   - Output exactly: **Verdict: Insufficient Evidence.** followed by a brief explanation.\n\n"
    "5. **CITATION RULES (STRICT):**\n"
    "   - **Valid Source Range:** You have access to Sources 1 through {max_id}. **ANY CITATION > {max_id} IS A HALLUCINATION.**\n"
    "   - **Atomic Citations:** Every specific claim must be cited immediately [N].\n"
    "   - **Verification:** Do not cite a source unless the text explicitly supports the claim.\n"
    "   - **The 'Eyes-Only' Rule:** IGNORE YOUR TRAINING DATA. Use ONLY facts present in the text chunks.\n"
    "### REQUIRED OUTPUT FORMAT:\n"
    "---------------------\n"
    f"{FEW_SHOT_EXAMPLE}\n"
    "---------------------\n\n"
    "### CONTEXT CHUNKS (Sources 1-{max_id}):\n"
    "{context_str}\n\n"
    "User Query: {query_str}\n"
    "Answer (using ONLY Sources 1-{max_id}):"
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
            context_str=context_str, query_str=query, max_id=RAG_RERANK_TOP_K
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
