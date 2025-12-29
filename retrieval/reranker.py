"""Cross-Encoder Reranker using ONNX Runtime for CPU efficiency.

This module implements a reranker that uses a quantized ONNX model
to re-score retrieval results. This provides SOTA accuracy (via mxbai-xsmall)
at CPU-friendly latencies (via INT8 quantization).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from llama_index.core.schema import NodeWithScore
from loguru import logger
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Path matches your setup_onnx.py output
DEFAULT_MODEL_PATH = "onnx_model_cache/mixedbread-ai_mxbai-rerank-xsmall-v1"


class CrossEncoderReranker:
    """ONNX-based cross-encoder reranker for improving retrieval relevance.

    Attributes:
        tokenizer: HuggingFace tokenizer.
        model: ONNX Runtime inference session.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
    ):
        """Initialize the ONNX reranker.

        Args:
            model_path: Path to the local quantized ONNX model directory.
            device: Device parameter (ignored for ONNX on CPU).
        """
        self.model_path = str(model_path)

        # Check if model exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                "Did you run 'python scripts/setup_onnx.py'?"
            )

        logger.info(f"Loading ONNX CrossEncoder from: {self.model_path}")
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load the quantized model explicitly
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path, file_name="model_quantized.onnx"
        )

        load_time = time.perf_counter() - start
        logger.info(f"ONNX Model loaded in {load_time:.3f}s")

    def predict(
        self, inputs: List[Tuple[str, str]], batch_size: int = 4
    ) -> List[float]:
        """Run inference on the ONNX model in batches to save memory.

        Args:
            inputs: List of (query, document) text pairs.
            batch_size: How many pairs to process at once. Lower = Less RAM.

        Returns:
            List of float scores between 0 and 1.
        """
        if not inputs:
            return []

        all_scores = []

        # Process in batches to prevent OOM (Out Of Memory) crashes
        # We loop from 0 to len(inputs) in steps of batch_size
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]

            # Tokenize only this small batch
            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Inference on this small batch
            with torch.no_grad():
                outputs = self.model(**features)
                logits = outputs.logits

                # Sigmoid activation
                batch_scores = 1 / (1 + np.exp(-logits.numpy()))

                # Append results
                all_scores.extend(batch_scores.flatten().tolist())

        return all_scores

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = 12,
        batch_size: int = 4,
    ) -> List[NodeWithScore]:
        """Re-rank nodes using ONNX cross-encoder scores."""
        if not nodes:
            logger.warning("Empty node list provided, returning empty results")
            return []

        logger.info(f"Reranking {len(nodes)} nodes for query: '{query[:50]}...'")
        start = time.perf_counter()

        # Create query-document pairs
        pairs = [(query, node.node.get_content()) for node in nodes]

        # Get scores via batched prediction
        # Batch size 4 is very safe for 8GB-16GB RAM
        scores = self.predict(pairs, batch_size=batch_size)

        # Create new NodeWithScore objects with updated scores
        scored_nodes = []
        for node, score in zip(nodes, scores):
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
