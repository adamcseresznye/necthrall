"""
Modular ProcessingAgent using Advanced RAG Components

This module implements a ProcessingAgent that uses the advanced modular RAG components
for section-aware document processing and enhanced embedding generation, while maintaining
full compatibility with the existing ProcessingAgent interface.

Core Features:
- Advanced section-aware document chunking with metadata
- High-performance batch embedding generation
- Hybrid retrieval combining BM25 and semantic search via RRF
- Cross-encoder reranking for final passage ranking
- Enhanced State fields: chunks, relevant_passages, processing_metadata
- Legacy compatibility: top_passages, processing_stats

Usage:
    agent = ProcessingAgent(app)  # From agents/processing.py (modular version)
    new_state = agent(state)

    # Enhanced fields (new):
    assert len(new_state.chunks) > 0  # Your Chunk objects
    assert len(new_state.relevant_passages) > 0  # Your enhanced Passage objects
    assert new_state.processing_metadata is not None  # Your ProcessingMetadata

    # Legacy compatibility (preserved):
    assert len(new_state.top_passages) == len(new_state.relevant_passages[:10])
"""

# Handle PyTorch DLL import issues (PyTorch conflicts with multithreading)
try:
    # Import in this specific order - matches conftest.py working pattern
    import fitz  # PyMuPDF first
    import torch  # PyTorch second

    torch.set_num_threads(1)  # Single-threaded mode to avoid DLL conflicts
    # Import CrossEncoder directly to ensure it's available
    from sentence_transformers import CrossEncoder

    print("✅ Pre-imported PyTorch and related libraries in safe order")
except Exception as e:
    print(f"⚠️  Could not pre-import PyTorch libraries: {e}")
    # Don't fail - let the agent attempt to load them later

import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple
from enum import Enum
from fastapi import FastAPI
import gc  # Added for explicit garbage collection

from models.state import (
    State,
    Chunk,
    Passage,
    ProcessingMetadata,
    RetrievalScores,
)
from rag.chunking import AdvancedDocumentChunker
from rag.embeddings import create_embedding_generator_from_app
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class ProcessingErrorType(str, Enum):
    """
    Enumeration of processing error types for systematic error handling and reporting.
    """

    OUT_OF_MEMORY = "out_of_memory"
    MODEL_LOADING = "model_loading"
    NETWORK_FAILURE = "network_failure"
    UNKNOWN = "unknown"


class ProcessingAgent:
    """
    Modular ProcessingAgent using Advanced RAG Components.

    Refactored to use AdvancedDocumentChunker and EmbeddingGenerator for enhanced
    section-aware processing while maintaining identical interface to legacy ProcessingAgent.

    Pipeline: Section Detection → Advanced Chunking → Batch Embedding → Hybrid Retrieval → Reranking

    Performance Targets:
    - Total processing time: <4s (matching legacy agent)
    - Memory usage: <500MB during processing
    - Chunks per paper: 50-100 typical academic papers
    """

    def __init__(self, app: FastAPI, embedder=None, chunker=None):
        """
        Initialize ProcessingAgent with modular RAG components.

        Args:
            app: FastAPI application with cached models in state
            embedder: Optional custom embedding generator (defaults to from app)
            chunker: Optional custom document chunker (defaults to AdvancedDocumentChunker)
        """
        self.app = app
        self.chunker = chunker if chunker is not None else AdvancedDocumentChunker()
        self.embedder = (
            embedder
            if embedder is not None
            else create_embedding_generator_from_app(app)
        )
        self.retriever = HybridRetriever()
        # Pass app for cached CrossEncoder
        self.reranker = CrossEncoderReranker(app=app)

        # Model warmup validation (matching legacy behavior)
        skip_warmup = embedder is not None  # Skip if custom embedder provided
        self._warmup_models(app, skip_warmup)

        # Stats for to_dict method (legacy compatibility)
        self._last_stats = None

        logger.info(
            "Modular ProcessingAgent initialized with AdvancedDocumentChunker and EmbeddingGenerator"
        )

    def __call__(self, state: State) -> State:
        """
        Synchronous wrapper for backward compatibility with existing tests and code.

        Args:
            state: Input State with filtered_papers, pdf_contents, config

        Returns:
            Enhanced State with chunks, relevant_passages, processing_metadata (plus legacy fields)
        """
        return self.process(state)

    def process(self, state: State) -> State:
        """
        Synchronous processing method using asyncio.run for backward compatibility.

        This allows existing synchronous code to work while maintaining async performance validation.

        Args:
            state: Input State with filtered_papers, pdf_contents, config

        Returns:
            Enhanced State with chunks, relevant_passages, processing_metadata (plus legacy fields)
        """
        return asyncio.run(self._aprocess(state))

    async def _aprocess(self, state: State) -> State:
        """
        Async version of processing pipeline for performance validation scripts.

        Args:
            state: Input State with filtered_papers, pdf_contents, config

        Returns:
            Enhanced State with chunks, relevant_passages, processing_metadata (plus legacy fields)
        """
        # Create copy for side-effect free operation
        new_state = state.model_copy()
        new_state.top_passages = []
        new_state.processing_stats = {}

        # Initialize enhanced fields
        new_state.chunks = []
        new_state.relevant_passages = []
        new_state.processing_metadata = ProcessingMetadata()

        # Get query and config
        query = state.optimized_query or state.original_query
        logger.info(
            f"Modular ProcessingAgent: Processing {len(state.filtered_papers)} papers for query: {query[:50]}..."
        )

        if not state.filtered_papers:
            logger.warning("Modular ProcessingAgent: No filtered papers to process")
            new_state.processing_stats = {
                "total_papers": 0,
                "error": "No filtered papers",
                "reason": "Empty filtered_papers list",
                "total_time": 0.0,
                "error_count": 1,
            }
            return new_state

        # Build PDF content lookup
        pdf_lookup = {pdf.paper_id: pdf for pdf in state.pdf_contents}

        # Initialize stats tracking (both legacy and enhanced)
        metadata = new_state.processing_metadata
        metadata.total_papers = len(state.filtered_papers)

        # Legacy stats structure (for backward compatibility)
        stats = {
            "total_papers": len(state.filtered_papers),
            "processed_papers": 0,
            "skipped_papers": 0,
            "total_sections": 0,
            "total_chunks": 0,
            "fallback_used_count": 0,
            "chunks_embedded": 0,
            "retrieval_candidates": 0,
            "reranked_passages": 0,
            "paper_errors": [],
            "stage_times": {
                "section_detection": 0.0,
                "chunking": 0.0,
                "embedding": 0.0,
                "index_build": 0.0,
                "retrieval": 0.0,
                "reranking": 0.0,
            },
            "total_time": 0.0,
        }

        total_start = time.perf_counter()

        try:
            # Stage 1: Process papers into chunks using AdvancedDocumentChunker
            chunk_start = time.perf_counter()
            chunks, errors, chunker_stats = self.chunker.process_papers(
                state.filtered_papers, state.pdf_contents
            )
            chunking_time = time.perf_counter() - chunk_start
            metadata.stage_times["chunking"] = chunking_time
            stats["stage_times"]["chunking"] = chunking_time
            logger.info(
                f"DEBUG: Chunking time set: {chunking_time:.3f}s, metadata.stage_times: {metadata.stage_times}"
            )

            new_state.chunks = chunks
            metadata.paper_errors.extend(errors)
            metadata.total_chunks = len(chunks)
            metadata.fallback_used_count = sum(
                1 for chunk in chunks if chunk.use_fallback
            )

            stats["total_sections"] = chunker_stats.get("section_distribution", {}).get(
                "total", 0
            )
            stats["total_chunks"] = len(chunks)
            stats["skipped_papers"] = len(errors)
            stats["fallback_used_count"] = metadata.fallback_used_count
            stats["processed_papers"] = len(state.filtered_papers) - len(errors)

            if not chunks:
                error_msg = "No chunks created from advanced document chunking"
                new_state.processing_stats = {
                    **stats,
                    "error": error_msg,
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            del chunker_stats  # Cleanup
            gc.collect()  # Explicit garbage collection

            # Stage 2: Generate embeddings using EmbeddingGenerator
            embed_start = time.perf_counter()
            chunk_dicts = [chunk.model_dump() for chunk in chunks]
            embedded_chunks = self.embedder.generate_embeddings_sync(
                chunk_dicts
            )  # Changed to sync call

            embedding_time = time.perf_counter() - embed_start
            metadata.chunks_embedded = len(embedded_chunks)
            metadata.stage_times["embedding"] = embedding_time
            logger.info(
                f"DEBUG: Embedding time set: {embedding_time:.3f}s, metadata.stage_times: {metadata.stage_times}"
            )

            stats["chunks_embedded"] = len(embedded_chunks)
            stats["stage_times"]["embedding"] = embedding_time

            if not embedded_chunks:
                error_msg = "No chunks successfully embedded"
                new_state.processing_stats = {
                    **stats,
                    "error": error_msg,
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            del chunk_dicts  # Cleanup
            gc.collect()  # Explicit garbage collection

            # Convert embeddings back to numpy arrays for the retriever
            import numpy as np

            for chunk in embedded_chunks:
                if "embedding" in chunk and isinstance(chunk["embedding"], list):
                    chunk["embedding"] = np.array(chunk["embedding"], dtype=np.float32)

            # Stage 3: Build retrieval index
            index_start = time.perf_counter()
            build_time_ms = self.retriever.build_indices(
                embedded_chunks, use_cache=False
            )
            build_time_sec = build_time_ms / 1000.0  # Convert ms to seconds
            metadata.stage_times["index_build"] = build_time_sec
            stats["stage_times"]["index_build"] = build_time_sec
            logger.info(
                f"DEBUG: Index build time set: {build_time_sec:.3f}s, metadata.stage_times: {metadata.stage_times}"
            )

            # Stage 4: Retrieve top candidates
            retrieve_start = time.perf_counter()
            candidates = self.retriever.retrieve(
                query, self.app.state.embedding_model.encode([query])[0], top_k=15
            )
            retrieval_time = time.perf_counter() - retrieve_start

            metadata.retrieval_candidates = len(candidates)
            metadata.stage_times["retrieval"] = retrieval_time
            stats["retrieval_candidates"] = len(candidates)
            stats["stage_times"]["retrieval"] = retrieval_time
            logger.info(
                f"DEBUG: Retrieval time set: {retrieval_time:.3f}s, metadata.stage_times: {metadata.stage_times}"
            )

            if not candidates:
                error_msg = "Hybrid retrieval returned no results"
                new_state.processing_stats = {
                    **stats,
                    "error": error_msg,
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            del embedded_chunks  # Cleanup
            gc.collect()  # Explicit garbage collection

            # Stage 5: Rerank to top 10
            rerank_start = time.perf_counter()
            reranked = self.reranker.rerank(query, candidates[:15])
            reranking_time = time.perf_counter() - rerank_start

            metadata.reranked_passages = len(reranked)
            metadata.stage_times["reranking"] = reranking_time
            stats["reranked_passages"] = len(reranked)
            stats["stage_times"]["reranking"] = reranking_time
            logger.info(
                f"DEBUG: Reranking time set: {reranking_time:.3f}s, metadata.stage_times: {metadata.stage_times}"
            )

            # Convert reranked results to enhanced Passage objects
            relevant_passages = []
            for passage_result in reranked[:10]:
                passage = Passage(
                    content=passage_result.get("content", ""),
                    section=passage_result.get("section", "other"),
                    paper_id=passage_result.get("paper_id", ""),
                    retrieval_score=passage_result.get("retrieval_score", 0.0),
                    cross_encoder_score=passage_result.get("cross_encoder_score"),
                    final_score=passage_result.get("final_score", 0.0),
                    scores=RetrievalScores(
                        semantic_score=passage_result.get("semantic_score"),
                        bm25_score=passage_result.get("bm25_score"),
                        final_rank=passage_result.get("rank", 0),
                    ),
                )
                relevant_passages.append(passage)

            new_state.relevant_passages = relevant_passages

            del candidates  # Cleanup
            gc.collect()  # Explicit garbage collection

            # Legacy compatibility: format top_passages
            top_passages = []
            for passage in relevant_passages:
                top_passages.append(
                    {
                        "content": passage.content,
                        "section": passage.section,
                        "paper_id": passage.paper_id,
                        "retrieval_score": passage.retrieval_score,
                        "cross_encoder_score": passage.cross_encoder_score,
                        "final_score": passage.final_score,
                    }
                )
            new_state.top_passages = top_passages

            # Finalize stats
            total_time = time.perf_counter() - total_start
            metadata.total_time = total_time
            metadata.processed_papers = stats["processed_papers"]

            stats["total_time"] = total_time
            new_state.processing_stats = stats

            # Store stats for to_dict method
            self._last_stats = stats
            self.export_performance_log(stats)

            logger.info(
                f"Modular ProcessingAgent: Completed in {total_time:.3f}s, "
                f"processed {metadata.processed_papers}/{metadata.total_papers} papers, "
                f"chunks {metadata.chunks_embedded}, relevant passages {len(relevant_passages)}"
            )

        except Exception as e:
            total_time = time.perf_counter() - total_start
            stats["total_time"] = total_time
            stats["error"] = str(e)
            metadata.processing_errors.append(str(e))
            new_state.processing_stats = stats
            new_state.top_passages = []
            new_state.relevant_passages = []
            new_state.chunks = []
            logger.error(f"Modular ProcessingAgent failed: {e}")
            raise

        return new_state

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        """
        Export processing stats as a JSON-serializable dictionary (legacy compatibility).
        """
        if not hasattr(self, "_last_stats") or self._last_stats is None:
            return {"error": "No processing stats available"}

        stats = self._last_stats.copy()
        stats["export_timestamp"] = time.time()
        stats["agent_version"] = "2.0.0"  # Modular version

        if minimal:
            stats.pop("paper_errors", None)
            stats.pop("stage_times", None)
            stats["export_type"] = "minimal"
        else:
            stats["export_type"] = "full"

        return stats

    def export_performance_log(
        self, stats: Dict[str, Any], log_file: str = "performance_log.json"
    ) -> None:
        """
        Export processing stats to performance log file (legacy compatibility).
        """
        try:
            export_data = self.to_dict(minimal=True)
            export_data["event"] = "modular_processing_agent_complete"
            export_data["timestamp"] = time.time()

            import json

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(export_data, ensure_ascii=False) + "\n")

            logger.info(
                f"Modular ProcessingAgent: Exported performance stats to {log_file}"
            )

        except Exception as e:
            logger.warning(
                f"Modular ProcessingAgent: Failed to export performance log: {e}"
            )

    def _warmup_models(self, app: FastAPI, skip_warmup: bool = False) -> None:
        """
        Warm up and validate cached models (matching legacy behavior).

        Args:
            app: FastAPI application with cached models
            skip_warmup: Skip validation if custom components are provided
        """
        if skip_warmup:
            logger.info(
                "Modular ProcessingAgent: Skipping model warmup (custom embedder provided)"
            )
            return

        try:
            warmup_start = time.perf_counter()

            if app is None or not hasattr(app, "state") or app.state is None:
                raise RuntimeError(
                    "App or app.state is None - cannot validate embedding model"
                )

            if (
                not hasattr(app.state, "embedding_model")
                or app.state.embedding_model is None
            ):
                raise RuntimeError("Embedding model not cached in app.state")

            test_embedding = app.state.embedding_model.encode(
                ["warmup test"], show_progress_bar=False
            )
            embedding_dim = len(test_embedding[0]) if len(test_embedding) > 0 else 384

            if embedding_dim != 384:
                raise RuntimeError(
                    f"Unexpected embedding dimensions: {embedding_dim}, expected 384"
                )

            warmup_time = time.perf_counter() - warmup_start
            logger.info(
                f"Modular ProcessingAgent: Models warmed up successfully in {warmup_time:.3f}s, "
                f"embedding_dim: {embedding_dim}"
            )

        except Exception as e:
            logger.error(f"Modular ProcessingAgent: Model warmup failed: {e}")
            raise RuntimeError(f"Model warmup failed: {e}") from e


# Alias for backward compatibility (tests may import ProcessingAgentV2)
ProcessingAgentV2 = ProcessingAgent
