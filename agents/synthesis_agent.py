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


CITATION_QA_TEMPLATE = (
    "You are an expert research synthesizer with a background in statistics and methodology. "
    "Your task is to generate a rigorous, scientifically accurate answer using ONLY the provided context passages.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "### INSTRUCTIONS (Follow Strictly):\n"
    "0. **Insufficient Information**: If the provided context does not contain the answer to the specific user query, state: 'The retrieved documents do not contain specific data regarding [topic].' Do not attempt to answer from general knowledge.\n\n"
    "1. **MANDATORY CITATIONS**: You MUST cite the source passage for EVERY factual claim using the exact index provided in the text (e.g., [1]).\n"
    "   - Place citations immediately after the specific data point (e.g., 'reduced by 20% [1]').\n"
    "   - If a sentence contains multiple claims, cite each one separately.\n"
    "   - **Do not output a single scientific statement without a citation.**\n\n"
    "2. **Methodological Priority**: \n"
    "   - Prioritize **Raw Data (Results)** and **Protocols (Methods)** over Abstract/Introduction summaries.\n"
    "   - If available, include specific dosages, sample sizes (N=), durations, or exact chemical compositions. Do not use vague terms like 'standard protocol' if the specific details are in the text.\n\n"
    "3. **Statistical Precision**: \n"
    "   - Never invent p-values. If a paper says '95% CI', do not write 'p=95'.\n"
    "   - Distinguish between 'No significant difference' (p>0.05) and 'Not tested'.\n"
    "   - If extracting numbers, include the unit (e.g., 'ms', 'mg/dL', '% change').\n\n"
    "4. **Contextual Nuance & Contradictions**: \n"
    "   - If a finding contradicts general consensus, you MUST add the condition (e.g., '...in short 5-minute tasks' or '...after caffeine intake').\n"
    "   - **Highlight Disagreements**: If [1] says 'Positive' and [2] says 'Negative', explicitly state: 'Evidence is mixed: [1] found positive effects, whereas [2] observed no change.'\n\n"
    "### REQUIRED OUTPUT FORMAT:\n"
    "You must use standard Markdown headers (###) and bullet points.\n\n"
    "### Executive Summary\n"
    "A high-level synthesis of the consensus and magnitude of effects. [1]\n\n"
    "### Key Findings\n"
    "* **Domain (e.g., Memory):** Synthesis of findings. Quantitative impact (e.g., 'reduced by 20%') [2].\n"
    "* **Domain (e.g., Attention):** Synthesis of findings. Nuance regarding task type or population [3].\n\n"
    "### Detailed Comparison (or Mechanism)\n"
    "Create a Markdown table comparing specific outcomes, dosages, or protocols if data allows.\n"
    "| Outcome/Variable | Study [1] Findings | Study [2] Findings |\n"
    "|---|---|---|\n"
    "| Metric 1 | Result (e.g., 50mg dose) [1] | Result (e.g., 100mg dose) [2] |\n\n"
    "### Limitations & Quality of Evidence\n"
    "Critique the quality of the data (e.g., 'Most studies were short-term', 'Small sample sizes', 'Lack of control group'). [4]\n\n"
    "---------------------\n"
    "User Query: {query_str}\n"
    "Scientific Answer: "
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
