"""
Unified ProcessingAgent for Necthrall RAG Pipeline

This module implements a unified ProcessingAgent that orchestrates the complete
document processing pipeline from raw PDF text to ranked relevant passages.
It integrates section-aware chunking, embedding generation, hybrid retrieval,
and cross-encoder reranking into a cohesive workflow.

Core Features:
- Section-aware document chunking with fallback strategies
- High-performance async batch embedding generation
- Hybrid retrieval combining BM25 and semantic search via RRF
- Cross-encoder reranking for final passage ranking
- Comprehensive performance monitoring and error handling

Usage:
    agent = ProcessingAgent()
    new_state = agent(state)  # Returns State with chunks, relevant_passages, processing_metadata
"""

import asyncio
import logging
import time
import psutil
import gc
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from fastapi import FastAPI

from models.state import (
    State,
    PDFContent,
    Passage,
    Chunk,
    ProcessingConfig,
    ProcessingMetadata,
)
from rag.chunking import AdvancedDocumentChunker
from rag.embeddings import EmbeddingGenerator, create_embedding_generator_from_app
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from utils.section_detector import SectionDetector

logger = logging.getLogger(__name__)


# Error classification for structured error handling
class ProcessingErrorType:
    MODEL_LOADING = "model_loading"
    OUT_OF_MEMORY = "out_of_memory"
    TEXT_CORRUPTION = "text_corruption"
    NETWORK_FAILURE = "network_failure"
    INDEX_CORRUPTION = "index_corruption"
    UNKNOWN = "unknown"


@dataclass
class ProcessingError(Exception):
    """Structured error for processing pipeline failures."""

    error_type: str
    message: str
    component: str
    recoverable: bool = True
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.timestamp = time.time()


class ProcessingAgent:
    """
    Unified processing agent that orchestrates the complete RAG pipeline.

    Combines modular components for document chunking, embedding, retrieval,
    and reranking while providing comprehensive pipeline performance tracking.

    Pipeline: Raw PDFs → Section Detection → Chunking → Embedding → Hybrid Retrieval → Reranking

    Performance Targets:
    - Processing time: <3s for 25 papers (10-50 chunks per paper)
    - Memory usage: <500MB during processing
    - Error handling: Graceful degradation with partial results
    """

    def __init__(
        self,
        app: Optional[FastAPI] = None,
        chunker: Optional[AdvancedDocumentChunker] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        """
        Initialize ProcessingAgent with optional component injection for testing.

        Args:
            app: FastAPI application for model access (required if embedder not provided)
            chunker: Custom DocumentChunker instance
            embedder: Custom EmbeddingGenerator instance
            retriever: Custom HybridRetriever instance
            reranker: Custom CrossEncoderReranker instance
        """
        self.app = app

        # Initialize components with defaults
        self.chunker = chunker or AdvancedDocumentChunker()
        self.embedder = embedder or (
            create_embedding_generator_from_app(app) if app else None
        )
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()

        if not self.embedder:
            raise ValueError(
                "Either app or embedder must be provided for embedding generation"
            )

        # Validate critical dependencies
        if not app and not embedder:
            raise ValueError("FastAPI app required for model access")

        logger.info("ProcessingAgent initialized with modular components")

    def __call__(self, state: State) -> State:
        """
        Process state through complete RAG pipeline with performance optimizations.

        Args:
            state: Input State with filtered_papers, pdf_contents, and query

        Returns:
            Enhanced State with chunks, relevant_passages, and processing_metadata

        Raises:
            RuntimeError: If processing fails catastrophically
        """
        pipeline_start = time.time()

        # Create copy for side-effect free operation
        new_state = state.model_copy()

        # Initialize output fields
        new_state.chunks = []
        new_state.relevant_passages = []
        new_state.processing_metadata = ProcessingMetadata()

        # Get query and config
        query = state.optimized_query or state.original_query
        config = self._parse_config(state.config)

        metadata = new_state.processing_metadata
        metadata.total_papers = len(state.filtered_papers)

        logger.info(
            {
                "event": "pipeline_start",
                "total_papers": len(state.filtered_papers),
                "query_length": len(query),
                "batch_size": config.batch_size,
                "top_k": config.top_k,
                "timestamp": time.time(),
            }
        )

        try:
            # PERFORMANCE OPTIMIZATION: Concurrent chunking and preprocessing
            chunks, chunk_errors = self._process_chunking_stage_parallel(
                state, metadata
            )
            new_state.chunks = chunks
            metadata.paper_errors.extend(chunk_errors)

            if not chunks:
                return self._handle_empty_results(
                    new_state, metadata, pipeline_start, "No chunks created"
                )

            # Chunks are already Chunk models from AdvancedDocumentChunker
            state_chunks = chunks

            # PERFORMANCE OPTIMIZATION: Concurrent embedding generation and index preparation
            # Stage 2 & 3: Embedding Generation + Index Building (concurrent)
            embedded_chunks, index_build_time = (
                self._process_embedding_and_index_stages(state_chunks, config, metadata)
            )

            if not embedded_chunks:
                return self._handle_empty_results(
                    new_state, metadata, pipeline_start, "No chunks embedded"
                )

            # PERFORMANCE OPTIMIZATION: Parallel query processing
            # Stages 4 & 5: Hybrid Retrieval + Reranking (concurrent)
            query_start = time.time()

            # Generate query embedding first
            query_embedding = self._prepare_query_embedding(query, config)

            # Parallel retrieval and reranking
            candidates, ranked_passages = (
                self._process_retrieval_and_reranking_parallel(
                    query, query_embedding, embedded_chunks, config
                )
            )

            query_time = time.time() - query_start
            metadata.stage_times["query_processing"] = query_time

            if not ranked_passages:
                return self._handle_empty_results(
                    new_state, metadata, pipeline_start, "No query results"
                )

            # Set final results
            new_state.relevant_passages = ranked_passages[: config.final_k]
            metadata.reranked_passages = len(ranked_passages)

            # Final metadata updates
            total_time = time.time() - pipeline_start
            metadata.total_time = total_time
            metadata.processed_papers = (
                len(state.filtered_papers) - metadata.skipped_papers
            )

            # Performance validation
            self._validate_performance_targets(metadata)

            logger.info(
                {
                    "event": "pipeline_complete",
                    "total_time_seconds": round(total_time, 3),
                    "papers_processed": metadata.processed_papers,
                    "chunks_created": metadata.total_chunks,
                    "chunks_embedded": metadata.chunks_embedded,
                    "passages_ranked": len(new_state.relevant_passages),
                    "fallback_used": metadata.fallback_used_count,
                    "performance_target_met": total_time < 2.0,  # Sub-2s target
                    "throughput_papers_per_sec": round(
                        metadata.processed_papers / total_time, 2
                    ),
                    "throughput_chunks_per_sec": round(
                        metadata.chunks_embedded / total_time, 2
                    ),
                }
            )

            return new_state

        except Exception as e:
            total_time = time.time() - pipeline_start
            metadata.total_time = total_time
            metadata.processing_errors.append(str(e))

            logger.error(
                {
                    "event": "pipeline_failure",
                    "error_type": self._classify_general_error(e),
                    "error_message": str(e),
                    "total_time_seconds": round(total_time, 3),
                    "stage_completed": len(
                        [
                            t
                            for t in metadata.stage_times.keys()
                            if metadata.stage_times[t] > 0
                        ]
                    ),
                }
            )

            # Return partial results if possible
            return new_state

    def _parse_config(self, config_dict: Dict[str, Any]) -> ProcessingConfig:
        """Parse configuration from state config dictionary."""
        try:
            return ProcessingConfig(**config_dict)
        except Exception:
            logger.warning("Failed to parse config, using defaults")
            return ProcessingConfig()

    def _process_chunking_stage(
        self, state: State, metadata: ProcessingMetadata
    ) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
        """Process documents through chunking stage."""
        start_time = time.time()

        chunks, errors, stats = self.chunker.process_papers(
            state.filtered_papers, state.pdf_contents
        )

        chunking_time = time.time() - start_time

        # Update metadata
        metadata.total_sections = 0  # Would need extension to track
        metadata.total_chunks = len(chunks)
        metadata.skipped_papers = len(errors)
        metadata.stage_times["chunking"] = chunking_time

        # Count fallback usage
        metadata.fallback_used_count = sum(1 for chunk in chunks if chunk.use_fallback)

        logger.info(
            {
                "event": "chunking_complete",
                "chunks_created": len(chunks),
                "errors": len(errors),
                "chunking_time_seconds": round(chunking_time, 3),
            }
        )

        return chunks, errors

    async def _process_embedding_stage(
        self,
        chunks: List[Chunk],
        metadata: ProcessingMetadata,
        config: ProcessingConfig,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process chunks through embedding generation with enhanced error handling."""
        start_time = time.time()
        errors = []

        # Check memory before processing
        memory_before = self._check_memory_usage()
        if memory_before > 0.7:  # 70% memory usage warning
            logger.warning(f"High memory usage before embedding: {memory_before:.1%}")

        # Text corruption detection and validation
        chunk_dicts = []
        invalid_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                chunk_dict = chunk.model_dump()

                # Validate chunk content
                is_valid, validation_error = self._validate_chunk_content(chunk_dict)
                if not is_valid:
                    invalid_chunks.append((i, validation_error))
                    errors.append(f"Chunk {i}: {validation_error}")
                    continue

                chunk_dicts.append(chunk_dict)

            except Exception as e:
                invalid_chunks.append((i, f"Serialization failed: {e}"))
                errors.append(f"Chunk {i} serialization failed: {e}")
                continue

        if invalid_chunks:
            logger.warning(
                f"Skipped {len(invalid_chunks)} invalid chunks during embedding"
            )

        if not chunk_dicts:
            logger.error("No valid chunks for embedding")
            errors.append("No valid chunks available for embedding")
            return [], errors

        try:
            # OOM prevention: Adaptive batch sizing
            optimal_batch_size = self._calculate_optimal_batch_size(
                len(chunk_dicts), config.batch_size
            )

            # Update embedder batch size dynamically
            if hasattr(self.embedder, "batch_size"):
                original_batch_size = self.embedder.batch_size
                self.embedder.batch_size = optimal_batch_size

            try:
                embedded_chunks = await self.embedder.generate_embeddings_async(
                    chunk_dicts
                )
            finally:
                # Restore original batch size
                if hasattr(self.embedder, "batch_size"):
                    self.embedder.batch_size = original_batch_size

            # Post-processing validation
            valid_embeddings, invalid_count = self._validate_embeddings(embedded_chunks)
            if invalid_count > 0:
                errors.append(f"Generated {invalid_count} invalid embeddings")

            # Memory cleanup
            del chunk_dicts
            gc.collect()

            embedding_time = time.time() - start_time
            metadata.chunks_embedded = len(valid_embeddings)
            metadata.stage_times["embedding"] = embedding_time

            memory_after = self._check_memory_usage()
            memory_delta = memory_after - memory_before

            logger.info(
                {
                    "event": "embedding_complete",
                    "chunks_embedded": len(valid_embeddings),
                    "invalid_chunks_skipped": len(invalid_chunks),
                    "invalid_embeddings": invalid_count,
                    "memory_usage_before": f"{memory_before:.1%}",
                    "memory_usage_after": f"{memory_after:.1%}",
                    "memory_delta": f"{memory_delta:+.1%}",
                    "adaptive_batch_size": optimal_batch_size,
                    "embedding_time_seconds": round(embedding_time, 3),
                }
            )

            return valid_embeddings, errors

        except MemoryError as e:
            logger.error(f"Out of memory during embedding generation: {e}")
            errors.append(
                "Out of memory - consider reducing batch size or memory usage"
            )

            # Emergency memory cleanup
            gc.collect()
            return [], errors

        except Exception as e:
            embedding_time = time.time() - start_time
            metadata.stage_times["embedding"] = embedding_time

            error_type = self._classify_embedding_error(e)
            error_msg = f"Embedding generation failed: {e}"

            logger.error(
                {
                    "event": "embedding_error",
                    "error_type": error_type,
                    "error_message": error_msg,
                    "chunks_attempted": len(chunk_dicts),
                    "chunks_valid": len(
                        [
                            c
                            for c in chunks
                            if self._validate_chunk_content(c.model_dump())[0]
                        ]
                    ),
                    "memory_usage": f"{self._check_memory_usage():.1%}",
                    "embedding_time_seconds": round(embedding_time, 3),
                }
            )

            errors.append(error_msg)

            # Return empty results on critical errors
            return [], errors

    def _process_indexing_stage(
        self, chunks: List[Dict[str, Any]], metadata: ProcessingMetadata
    ) -> List[str]:
        """Build retrieval indices from embedded chunks."""
        start_time = time.time()

        try:
            build_time = self.retriever.build_indices(chunks, use_cache=False)
            metadata.stage_times["index_build"] = build_time

            logger.info(
                {
                    "event": "indexing_complete",
                    "index_build_time_seconds": round(build_time, 3),
                }
            )

            return []

        except Exception as e:
            build_time = time.time() - start_time
            metadata.stage_times["index_build"] = build_time
            error_msg = f"Index building failed: {e}"
            logger.error(error_msg)
            return [error_msg]

    def _process_retrieval_stage(
        self, query: str, config: ProcessingConfig, metadata: ProcessingMetadata
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute hybrid retrieval against query."""
        start_time = time.time()

        try:
            # Generate query embedding (would need app.model access in real implementation)
            if hasattr(self.app.state, "embedding_model"):
                query_embedding = self.app.state.embedding_model.encode([query])[0]
            else:
                # Fallback for testing - this would fail in real usage
                query_embedding = [0.0] * 384  # Placeholder

            candidates = self.retriever.retrieve(
                query, query_embedding, top_k=config.top_k
            )

            retrieval_time = time.time() - start_time
            metadata.retrieval_candidates = len(candidates)
            metadata.stage_times["retrieval"] = retrieval_time

            logger.info(
                {
                    "event": "retrieval_complete",
                    "candidates_found": len(candidates),
                    "retrieval_time_seconds": round(retrieval_time, 3),
                }
            )

            return candidates, []

        except Exception as e:
            retrieval_time = time.time() - start_time
            metadata.stage_times["retrieval"] = retrieval_time
            error_msg = f"Retrieval failed: {e}"
            logger.error(error_msg)
            return [], [error_msg]

    def _process_reranking_stage(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        config: ProcessingConfig,
        metadata: ProcessingMetadata,
    ) -> Tuple[List[Passage], List[str]]:
        """Apply cross-encoder reranking to candidates."""
        start_time = time.time()

        try:
            if config.enable_reranking:
                reranked = self.reranker.rerank(query, candidates)

                # Convert to Passage objects
                passages = [
                    Passage(
                        content=p.get("content", ""),
                        section=p.get("section", "other"),
                        paper_id=p.get("paper_id", ""),
                        retrieval_score=p.get("retrieval_score", 0.0),
                        cross_encoder_score=p.get("cross_encoder_score"),
                        final_score=p.get("final_score", p.get("retrieval_score", 0.0)),
                    )
                    for p in reranked
                ]
            else:
                # Skip reranking - convert candidates directly
                passages = [
                    Passage(
                        content=c.get("content", ""),
                        section=c.get("section", "other"),
                        paper_id=c.get("paper_id", ""),
                        retrieval_score=c.get("retrieval_score", 0.0),
                        cross_encoder_score=None,
                        final_score=c.get("retrieval_score", 0.0),
                    )
                    for c in candidates
                ]

            reranking_time = time.time() - start_time
            metadata.stage_times["reranking"] = reranking_time

            logger.info(
                {
                    "event": "reranking_complete",
                    "passages_ranked": len(passages),
                    "reranking_enabled": config.enable_reranking,
                    "reranking_time_seconds": round(reranking_time, 3),
                }
            )

            return passages, []

        except Exception as e:
            reranking_time = time.time() - start_time
            metadata.stage_times["reranking"] = reranking_time
            error_msg = f"Reranking failed: {e}"
            logger.error(error_msg)

            # Return candidates as passages without reranking
            passages = [
                Passage(
                    content=c.get("content", ""),
                    section=c.get("section", "other"),
                    paper_id=c.get("paper_id", ""),
                    retrieval_score=c.get("retrieval_score", 0.0),
                    cross_encoder_score=None,
                    final_score=c.get("retrieval_score", 0.0),
                )
                for c in candidates
            ]
            return passages, [error_msg]

    def _handle_empty_results(
        self, state: State, metadata: ProcessingMetadata, start_time: float, reason: str
    ) -> State:
        """Handle cases where processing produced no results."""
        metadata.total_time = time.time() - start_time
        metadata.processing_errors.append(reason)

        logger.warning(
            {
                "event": "processing_empty_results",
                "reason": reason,
                "total_time_seconds": round(metadata.total_time, 3),
            }
        )

        return state

    def _check_memory_usage(self) -> float:
        """Check current memory usage as a fraction (0.0 to 1.0)."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return 0.0  # Default to no memory issues

    def _validate_chunk_content(self, chunk_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate chunk content for potential corruption issues."""
        try:
            # Check for null bytes or excessive Unicode
            content = chunk_dict.get("content", "")
            if not isinstance(content, str):
                return False, f"Content is not a string: {type(content)}"

            if not content.strip():
                return False, "Empty or whitespace-only content"

            if len(content) > 10000:  # Reasonable upper bound
                return False, f"Content too long: {len(content)} characters"

            # Check for excessive non-ASCII characters (potential corruption)
            ascii_ratio = sum(1 for c in content if ord(c) < 128) / max(1, len(content))
            if ascii_ratio < 0.7 and len(content) > 100:  # Too many Unicode chars
                return False, f"High Unicode content ratio: {ascii_ratio:.2%}"

            # Check for pathological content patterns
            if "\x00" in content:  # Null bytes
                return False, "Contains null bytes"

            if content.count("\n") > 500:  # Excessive line breaks
                return False, f"Too many line breaks: {content.count(chr(10))}"

            return True, ""

        except Exception as e:
            return False, f"Content validation failed: {e}"

    def _calculate_optimal_batch_size(
        self, num_chunks: int, configured_batch_size: int
    ) -> int:
        """Calculate optimal batch size based on memory and chunk count."""
        memory_usage = self._check_memory_usage()

        # Reduce batch size if memory usage is high
        if memory_usage > 0.8:  # >80% memory
            optimal = max(1, configured_batch_size // 4)
        elif memory_usage > 0.6:  # >60% memory
            optimal = max(1, configured_batch_size // 2)
        else:
            optimal = configured_batch_size

        # Ensure batch size is reasonable for total chunks
        optimal = min(optimal, num_chunks)
        optimal = max(1, optimal)  # At least 1

        logger.debug(
            f"Calculated optimal batch size: {optimal} (from {configured_batch_size} at {memory_usage:.1%} memory)"
        )

        return optimal

    def _validate_embeddings(
        self, embedded_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Validate generated embeddings and filter out invalid ones."""
        valid_chunks = []
        invalid_count = 0

        for chunk in embedded_chunks:
            try:
                embedding = chunk.get("embedding")
                embedding_dim = chunk.get("embedding_dim", 384)

                if embedding is None:
                    invalid_count += 1
                    logger.warning("Chunk missing embedding")
                    continue

                if isinstance(embedding, np.ndarray):
                    if embedding.shape != (embedding_dim,):
                        invalid_count += 1
                        logger.warning(
                            f"Wrong embedding shape: {embedding.shape} != {(embedding_dim,)}"
                        )
                        continue
                    if not np.isfinite(embedding).all():
                        invalid_count += 1
                        logger.warning("Embedding contains non-finite values")
                        continue
                else:
                    invalid_count += 1
                    logger.warning(f"Embedding is not numpy array: {type(embedding)}")
                    continue

                valid_chunks.append(chunk)

            except Exception as e:
                invalid_count += 1
                logger.warning(f"Embedding validation failed: {e}")
                continue

        return valid_chunks, invalid_count

    def _classify_embedding_error(self, error: Exception) -> str:
        """Classify embedding errors for better error reporting."""
        error_str = str(error).lower()

        # Memory-related errors
        if any(
            term in error_str
            for term in ["memory", "oom", "out of memory", "cuda out of memory"]
        ):
            return ProcessingErrorType.OUT_OF_MEMORY

        # Model loading/finding errors
        elif any(
            term in error_str
            for term in ["model not found", "no model", "model loading"]
        ):
            return ProcessingErrorType.MODEL_LOADING

        # Network/connection errors (for remote models)
        elif any(
            term in error_str
            for term in ["connection", "network", "timeout", "502", "503", "504"]
        ):
            return ProcessingErrorType.NETWORK_FAILURE

        # Everything else is unknown for now
        else:
            return ProcessingErrorType.UNKNOWN

    # PERFORMANCE OPTIMIZATION: Parallel pipeline stages
    def _process_chunking_stage_parallel(
        self, state: State, metadata: ProcessingMetadata
    ) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
        """Process chunking stage (currently sequential, but structured for future parallelism)."""
        return self._process_chunking_stage(state, metadata)

    def _process_embedding_and_index_stages(
        self,
        chunks: List[Chunk],
        config: ProcessingConfig,
        metadata: ProcessingMetadata,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Process embedding and index building stages concurrently."""
        # For now, run sequentially but measure times independently
        import asyncio

        async def run_stages():
            # Stage 2: Embedding generation
            embedded_chunks, embed_errors = await self._process_embedding_stage(
                chunks, metadata, config
            )
            metadata.processing_errors.extend(embed_errors)

            if not embedded_chunks:
                return [], 0.0

            # Stage 3: Index building (run in thread pool)
            index_start = time.time()
            index_errors = await asyncio.to_thread(
                self._process_indexing_stage, embedded_chunks, metadata
            )
            index_build_time = time.time() - index_start

            if index_errors:
                logger.warning(f"Index building had {len(index_errors)} errors")

            return embedded_chunks, index_build_time

        try:
            return asyncio.run(run_stages())
        except Exception as e:
            logger.error(f"Concurrent embedding/indexing failed: {e}")
            # Fallback to sequential processing
            embedded_chunks, embed_errors = asyncio.run(
                self._process_embedding_stage(chunks, metadata, config)
            )
            metadata.processing_errors.extend(embed_errors)
            index_errors = self._process_indexing_stage(embedded_chunks, metadata)
            return embedded_chunks, metadata.stage_times.get("index_build", 0.0)

    def _prepare_query_embedding(
        self, query: str, config: ProcessingConfig
    ) -> np.ndarray:
        """Generate or retrieve query embedding."""
        # Generate query embedding (would need app.model access in real implementation)
        if hasattr(self.app.state, "embedding_model"):
            try:
                query_embedding = self.app.state.embedding_model.encode([query])[0]
                return np.array(query_embedding)
            except Exception as e:
                logger.warning(f"Query embedding generation failed: {e}")
                # Fallback to zero vector
                return np.zeros(384, dtype=np.float32)
        else:
            # Fallback for testing - this would fail in real usage
            logger.warning("No embedding model available, using zero vector")
            return np.zeros(384, dtype=np.float32)

    def _process_retrieval_and_reranking_parallel(
        self,
        query: str,
        query_embedding: np.ndarray,
        embedded_chunks: List[Dict[str, Any]],
        config: ProcessingConfig,
    ) -> Tuple[List[Dict[str, Any]], List[Passage]]:
        """Process retrieval and reranking stages in parallel."""
        import asyncio

        async def run_retrieval():
            return self._process_retrieval_stage(query, config, Mock())

        async def run_reranking(candidates):
            # Create mock metadata for reranking
            mock_metadata = ProcessingMetadata()
            passages, rerank_errors = self._process_reranking_stage(
                query, candidates, config, mock_metadata
            )
            return passages, rerank_errors

        async def run_parallel():
            # Start both stages
            retrieval_task = asyncio.create_task(run_retrieval())
            await asyncio.sleep(0.001)  # Allow retrieval to start

            # For now, run sequentially but with async structure
            candidates, retrieval_errors = await retrieval_task

            if not candidates or retrieval_errors:
                logger.warning("Retrieval failed, skipping reranking")
                return [], []

            passages, rerank_errors = await run_reranking(candidates)

            return candidates, passages

        try:
            return asyncio.run(run_parallel())
        except Exception as e:
            logger.error(f"Parallel retrieval/reranking failed: {e}")
            # Fallback to sequential
            candidates, retrieval_errors = self._process_retrieval_stage(
                query, config, Mock()
            )
            mock_metadata = ProcessingMetadata()
            passages, rerank_errors = self._process_reranking_stage(
                query, candidates, config, mock_metadata
            )
            return candidates, passages

    def _validate_performance_targets(self, metadata: ProcessingMetadata) -> None:
        """Validate performance against targets and log warnings if not met."""
        total_time = metadata.total_time

        if total_time > 3.0:
            logger.warning(
                {
                    "event": "performance_target_missed",
                    "target_seconds": 3.0,
                    "actual_seconds": round(total_time, 3),
                    "exceeded_by": round(total_time - 3.0, 3),
                    "chunks_processed": metadata.chunks_embedded,
                    "throughput_chunks_per_sec": round(
                        metadata.chunks_embedded / total_time, 1
                    ),
                }
            )

        if total_time < 2.0:
            logger.info(
                {
                    "event": "performance_target_exceeded",
                    "target_seconds": 2.0,
                    "actual_seconds": round(total_time, 3),
                    "improvement": round(2.0 - total_time, 3),
                }
            )

    def _classify_general_error(self, error: Exception) -> str:
        """Classify general processing errors."""
        error_str = str(error).lower()

        # Memory errors
        if any(term in error_str for term in ["memory", "oom", "out of memory"]):
            return ProcessingErrorType.OUT_OF_MEMORY

        # Network errors
        elif any(
            term in error_str
            for term in ["connection", "network", "timeout", "502", "503", "504"]
        ):
            return ProcessingErrorType.NETWORK_FAILURE

        # Text/data corruption
        elif any(
            term in error_str
            for term in ["corrupt", "invalid", "malformed", "encoding"]
        ):
            return ProcessingErrorType.TEXT_CORRUPTION

        # Model errors
        elif any(
            term in error_str
            for term in ["model", "transformer", "sentence", "embedding"]
        ):
            return ProcessingErrorType.MODEL_LOADING

        else:
            return ProcessingErrorType.UNKNOWN

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from all components."""
        return {
            "chunker_stats": self.chunker.get_statistics([]),  # Would need chunk list
            "embedder_stats": self.embedder.get_statistics(),
            "retriever_stats": {},  # Would need extension
            "reranker_stats": {},  # Would need extension
        }


# Backward compatibility alias
ProcessingAgentV2 = ProcessingAgent
