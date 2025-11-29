"""Cross-Encoder Reranker for improving retrieval relevance.

This module implements a reranker that uses a cross-encoder model to
re-score retrieval results for improved relevance ranking.

Cross-encoders process query-document pairs together, allowing for
richer semantic understanding compared to bi-encoder approaches.

Usage:
    from retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query="fasting benefits", nodes=retrieval_results)

Performance:
    - Re-ranking 15 items: <600ms on CPU
    - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (lightweight, fast)
    - CPU-only: No GPU dependencies

Note:
    On Windows, torch must be imported before sentence_transformers to avoid
    DLL initialization errors. This module handles this automatically.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, TYPE_CHECKING

from loguru import logger

# WINDOWS DLL FIX: Set up torch DLL path before importing sentence_transformers
# This prevents DLL initialization errors when onnxruntime is also used
if os.name == "nt":
    try:
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# CRITICAL: Import torch BEFORE any library that might load onnxruntime
# (including llama_index and sentence_transformers) to avoid DLL conflicts on Windows
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Now safe to import llama_index (which may load onnxruntime)
from llama_index.core.schema import NodeWithScore, TextNode

# Now safe to import CrossEncoder
from sentence_transformers import CrossEncoder


# Default model - lightweight and fast for CPU inference
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Cross-encoder based reranker for improving retrieval relevance.

    This class takes retrieval results and re-scores them using a cross-encoder
    model, which processes query-document pairs together for better semantic
    matching.

    Attributes:
        model_name: Name of the cross-encoder model.
        model: The loaded CrossEncoder instance.

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> results = reranker.rerank(
        ...     query="What is fasting?",
        ...     nodes=retriever_results,
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
            device: Device to run inference on. Default: 'cpu'.
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading CrossEncoder model: {model_name} on {device}")
        start = time.perf_counter()

        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device=device,
        )

        load_time = time.perf_counter() - start
        logger.info(f"CrossEncoder loaded in {load_time:.3f}s")

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = 12,
    ) -> List[NodeWithScore]:
        """Re-rank nodes using cross-encoder scores.

        Args:
            query: The search query.
            nodes: List of NodeWithScore from retriever.
            top_k: Number of top results to return (default: 12).

        Returns:
            List of NodeWithScore sorted by cross-encoder score (descending).
            The scores are replaced with cross-encoder scores.
        """
        # Handle empty input
        if not nodes:
            logger.warning("Empty node list provided, returning empty results")
            return []

        logger.info(f"Reranking {len(nodes)} nodes for query: '{query[:50]}...'")
        start = time.perf_counter()

        # Create query-document pairs for cross-encoder
        pairs = [(query, node.node.get_content()) for node in nodes]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Create new NodeWithScore objects with updated scores
        scored_nodes = []
        for node, score in zip(nodes, scores):
            # Create a new NodeWithScore with the cross-encoder score
            new_node = NodeWithScore(
                node=node.node,
                score=float(score),
            )
            scored_nodes.append(new_node)

        # Sort by score descending
        scored_nodes.sort(key=lambda x: x.score, reverse=True)

        # Limit to top_k
        results = scored_nodes[:top_k]

        elapsed = time.perf_counter() - start
        logger.info(f"Reranking completed: {len(results)} results in {elapsed:.3f}s")

        return results
