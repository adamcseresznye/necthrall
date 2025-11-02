"""
Embedding Generation Module for Necthrall RAG System

This module handles high-performance batch embedding generation for document chunks,
optimized for CPU processing with async support and memory-efficient batching.
"""

import asyncio
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import json

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    High-performance CPU-based embedding generator for RAG document processing.

    Processes document chunks into high-dimensional embeddings using SentenceTransformers,
    optimized for batch processing with async support and memory-efficient execution.

    Core Features:
    - Async batch processing with CPU thread pooling
    - Automatic embedding validation and error recovery
    - Memory usage monitoring and optimization
    - Configurable batch sizes for different deployment scales

    Performance Targets:
    - Processing rate: 10,000+ chunks per minute on modern CPUs
    - Memory efficiency: <400MB peak usage for large batches
    - Reliability: Graceful handling of individual chunk failures
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        embedding_dim: int = 384,
        batch_size: int = 64,
    ):
        """
        Initialize EmbeddingGenerator with SentenceTransformer model.

        Args:
            embedding_model: Pre-loaded SentenceTransformer instance
            embedding_dim: Expected embedding dimensions (default: 384 for all-MiniLM-L6-v2)
            batch_size: Number of chunks to process per batch (default: 32)
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # Validate model on initialization
        self._validate_model()

        # Performance tracking
        self.total_processed = 0
        self.total_batches = 0
        self.total_time = 0.0

        logger.info(
            json.dumps(
                {
                    "event": "embedding_generator_initialized",
                    "embedding_dim": self.embedding_dim,
                    "batch_size": self.batch_size,
                    "model_type": type(self.embedding_model).__name__,
                }
            )
        )

    def _validate_model(self) -> None:
        """
        Validate that the embedding model is properly configured.

        Raises:
            RuntimeError: If model is invalid or produces unexpected dimensions
        """
        if not isinstance(self.embedding_model, SentenceTransformer):
            raise RuntimeError(f"Invalid model type: {type(self.embedding_model)}")

        try:
            # Quick test embedding
            test_text = ["test embedding validation"]
            test_result = self.embedding_model.encode(
                test_text, show_progress_bar=False, convert_to_numpy=True
            )

            if test_result.shape != (1, self.embedding_dim):
                raise RuntimeError(
                    f"Model produces {test_result.shape[1]} dimensions, expected {self.embedding_dim}"
                )

        except Exception as e:
            raise RuntimeError(f"Model validation failed: {e}") from e

    def generate_embeddings_sync(self, chunks: List[Dict]) -> List[Dict]:
        stage_start = time.perf_counter()

        # Extract all texts at once
        texts = [chunk["content"] for chunk in chunks]

        # Single batch call - let sentence-transformers optimize internally
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
            device="cpu",  # Explicit CPU specification (no GPU in production)
        )

        # Assign embeddings
        result_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding.tolist()
            chunk_with_embedding["embedding_dim"] = (
                self.embedding_dim
            )  # Add embedding_dim
            result_chunks.append(chunk_with_embedding)

        # Immediate cleanup
        del texts, embeddings
        import gc

        gc.collect()

        stage_time = time.perf_counter() - stage_start
        logger.info(f"EMBEDDING_STAGE: {len(chunks)} chunks in {stage_time:.3f}s")

        return result_chunks

    def _validate_results(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate processed chunks and filter out invalid ones.

        Args:
            chunks: List of chunks with embeddings added

        Returns:
            List of valid chunks with proper embeddings
        """
        valid_chunks = []

        for i, chunk in enumerate(chunks):
            if "embedding" not in chunk or "embedding_dim" not in chunk:
                logger.warning(f"Chunk {i}: missing embedding fields, skipping")
                continue

            embedding = chunk["embedding"]
            embedding_dim = chunk["embedding_dim"]

            if not isinstance(
                embedding, (np.ndarray, list)
            ):  # Accept both numpy array and list
                logger.warning(
                    f"Chunk {i}: embedding not numpy array or list, skipping"
                )
                continue

            if isinstance(
                embedding, list
            ):  # Convert list to numpy array for validation
                embedding = np.array(embedding, dtype=np.float32)

            if embedding.shape != (self.embedding_dim,):
                logger.warning(
                    f"Chunk {i}: wrong embedding shape {embedding.shape}, skipping"
                )
                continue

            if embedding_dim != self.embedding_dim:
                logger.warning(
                    f"Chunk {i}: wrong embedding_dim {embedding_dim}, skipping"
                )
                continue

            # Check for placeholder embeddings (all zeros)
            if np.allclose(embedding, 0.0):
                logger.warning(f"Chunk {i}: placeholder embedding detected")

            valid_chunks.append(chunk)

        if len(valid_chunks) != len(chunks):
            logger.warning(
                f"Embedding validation: {len(valid_chunks)}/{len(chunks)} chunks valid"
            )

        return valid_chunks

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for this generator instance.

        Returns:
            Dictionary with performance metrics
        """
        if self.total_batches == 0:
            return {"total_batches": 0}

        avg_time_per_batch = self.total_time / self.total_batches
        overall_throughput = (
            self.total_processed / self.total_time if self.total_time > 0 else 0
        )

        return {
            "total_chunks_processed": self.total_processed,
            "total_batches_processed": self.total_batches,
            "total_processing_time_seconds": round(self.total_time, 3),
            "avg_time_per_batch_seconds": round(avg_time_per_batch, 3),
            "overall_throughput_chunks_per_second": round(overall_throughput, 1),
            "batch_size": self.batch_size,
            "embedding_dimensions": self.embedding_dim,
        }

    def estimate_memory_usage(self, num_chunks: int) -> float:
        """
        Estimate memory usage for processing a given number of chunks.

        Args:
            num_chunks: Number of chunks to process

        Returns:
            Estimated memory usage in MB
        """
        # Embedding memory (float32 = 4 bytes per float) + overhead
        embedding_bytes = num_chunks * self.embedding_dim * 4
        total_mb = (embedding_bytes * 2.5) / (1024 * 1024)  # 2.5x overhead factor

        return round(total_mb, 1)


# Factory function for backward compatibility
def create_embedding_generator_from_app(
    app: FastAPI, batch_size: int = 64
) -> EmbeddingGenerator:
    """
    Create EmbeddingGenerator from FastAPI app with cached model.

    Args:
        app: FastAPI application with cached embedding model
        batch_size: Batch size for processing

    Returns:
        Configured EmbeddingGenerator instance

    Raises:
        RuntimeError: If model not available in app state
    """
    if not hasattr(app.state, "embedding_model") or app.state.embedding_model is None:
        raise RuntimeError("Embedding model not cached in app.state")

    model = app.state.embedding_model
    return EmbeddingGenerator(model, batch_size=batch_size)
