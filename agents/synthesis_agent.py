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

import re
from typing import List

from llama_index.core.schema import NodeWithScore
from loguru import logger

from config.config import get_settings
from config.prompts import CITATION_QA_TEMPLATE, FEW_SHOT_EXAMPLE
from utils.llm_router import LLMRouter


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

    def __init__(self, settings: "Settings" = None, temperature: float = 0.15) -> None:
        """Initialize the synthesis agent.

        Args:
            settings: Optional settings object. If None, loads from config.
            temperature: LLM temperature for generation (0.15 = focused but not rigid).
        """
        self.settings = settings or get_settings()
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
            max_id=get_settings().RAG_RERANK_TOP_K,
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
