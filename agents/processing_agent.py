import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI

from models.state import State, PDFContent
from utils.section_detector import SectionDetector
from utils.embedding_manager import EmbeddingManager
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class ProcessingAgent:
    """
    Processes filtered papers through section detection, chunking, embedding, hybrid retrieval, and reranking.

    Wraps the Week 2 pipeline into a callable agent that consumes filtered_papers from State
    and outputs top_passages and processing_stats. Side-effect free with respect to input State.

    Pipeline: section detect → chunk → embed → hybrid retrieve → rerank

    Usage example:
        agent = ProcessingAgent(app)
        new_state = agent(state)  # Returns State with top_passages and processing_stats
    """

    def __init__(self, app: FastAPI):
        """
        Initialize ProcessingAgent with FastAPI app for model access.

        Args:
            app: FastAPI application with cached models in state
        """
        self.app = app
        self.section_detector = SectionDetector()
        self.embedding_manager = EmbeddingManager(app, batch_size=32)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()

        # Stats for to_dict method
        self._last_stats = None

        # Model and tokenizer warmup validation
        self._warmup_models(app)

    def __call__(self, state: State) -> State:
        """
        Process filtered papers to extract top passages with scores.

        Args:
            state: Input State with filtered_papers and pdf_contents

        Returns:
            New State with top_passages and processing_stats populated (side-effect free)
        """
        # Create copy for side-effect free operation
        new_state = state.model_copy()
        new_state.top_passages = []
        new_state.processing_stats = {}

        # Initialize query
        query = state.optimized_query or state.original_query
        logger.info(
            f"ProcessingAgent: Processing {len(state.filtered_papers)} papers for query: {query[:50]}..."
        )

        if not state.filtered_papers:
            logger.warning("ProcessingAgent: No filtered papers to process")
            new_state.processing_stats = {
                "total_papers": 0,
                "error": "No filtered papers",
                "reason": "Empty filtered_papers list",
                "total_time": 0.0,
            }
            return new_state

        # Build PDF content lookup by paper_id
        pdf_lookup = {pdf.paper_id: pdf for pdf in state.pdf_contents}

        # Initialize processing stats
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
            "paper_errors": [],  # Per-paper error accumulation
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
            # Stage 1: Process papers into chunks
            all_chunks, paper_stats = self._process_papers_to_chunks(
                state.filtered_papers, pdf_lookup, stats
            )

            if not all_chunks:
                new_state.processing_stats = {
                    **stats,
                    "error": "No chunks created",
                    "reason": "All papers failed processing or produced no chunks",
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            # Stage 2: Embed all chunks
            try:
                embed_start = time.perf_counter()
                embedded_chunks = asyncio.run(
                    self.embedding_manager.process_chunks_async(all_chunks)
                )
                stats["stage_times"]["embedding"] = time.perf_counter() - embed_start
                stats["chunks_embedded"] = len(embedded_chunks)

                if not embedded_chunks:
                    error_info = {
                        "stage": "embedding",
                        "error_type": "EmptyEmbeddings",
                        "message": "No chunks successfully embedded",
                        "timestamp": time.time(),
                    }
                    stats["paper_errors"].append(error_info)
                    logger.error(f"ProcessingAgent: {error_info['message']}")
                    new_state.processing_stats = {
                        **stats,
                        "error": "Embedding failed",
                        "reason": "No chunks successfully embedded",
                        "total_time": time.perf_counter() - total_start,
                    }
                    new_state.top_passages = []
                    return new_state

            except Exception as e:
                error_info = {
                    "stage": "embedding",
                    "error_type": type(e).__name__,
                    "message": f"Embedding stage failed: {str(e)}",
                    "timestamp": time.time(),
                }
                stats["paper_errors"].append(error_info)
                logger.error(f"ProcessingAgent: {error_info['message']}")
                new_state.processing_stats = {
                    **stats,
                    "error": f"Embedding failed: {str(e)}",
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            # Stage 3: Build retrieval index
            try:
                index_start = time.perf_counter()
                build_time = self.hybrid_retriever.build_indices(
                    embedded_chunks, use_cache=False
                )
                stats["stage_times"]["index_build"] = build_time

            except Exception as e:
                error_info = {
                    "stage": "index_build",
                    "error_type": type(e).__name__,
                    "message": f"Index building failed: {str(e)}",
                    "timestamp": time.time(),
                }
                stats["paper_errors"].append(error_info)
                logger.error(f"ProcessingAgent: {error_info['message']}")
                new_state.processing_stats = {
                    **stats,
                    "error": f"Index building failed: {str(e)}",
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            # Stage 4: Retrieve top candidates
            try:
                retrieve_start = time.perf_counter()
                query_embedding = self.app.state.embedding_model.encode([query])[0]
                candidates = self.hybrid_retriever.retrieve(
                    query, query_embedding, top_k=15
                )
                stats["stage_times"]["retrieval"] = time.perf_counter() - retrieve_start
                stats["retrieval_candidates"] = len(candidates)

                if not candidates:
                    error_info = {
                        "stage": "retrieval",
                        "error_type": "EmptyCandidates",
                        "message": "Hybrid retrieval returned no results",
                        "timestamp": time.time(),
                    }
                    stats["paper_errors"].append(error_info)
                    logger.warning(f"ProcessingAgent: {error_info['message']}")
                    new_state.processing_stats = {
                        **stats,
                        "error": "No retrieval candidates",
                        "reason": "Hybrid retrieval returned no results",
                        "total_time": time.perf_counter() - total_start,
                    }
                    new_state.top_passages = []
                    return new_state

            except Exception as e:
                error_info = {
                    "stage": "retrieval",
                    "error_type": type(e).__name__,
                    "message": f"Retrieval failed: {str(e)}",
                    "timestamp": time.time(),
                }
                stats["paper_errors"].append(error_info)
                logger.error(f"ProcessingAgent: {error_info['message']}")
                new_state.processing_stats = {
                    **stats,
                    "error": f"Retrieval failed: {str(e)}",
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            # Stage 5: Rerank to top 10
            try:
                rerank_start = time.perf_counter()
                reranked = self.reranker.rerank(
                    query, candidates[:15]
                )  # Top-15 → top-10
                stats["stage_times"]["reranking"] = time.perf_counter() - rerank_start
                stats["reranked_passages"] = len(reranked)

            except Exception as e:
                error_info = {
                    "stage": "reranking",
                    "error_type": type(e).__name__,
                    "message": f"Reranking failed: {str(e)}",
                    "timestamp": time.time(),
                }
                stats["paper_errors"].append(error_info)
                logger.error(f"ProcessingAgent: {error_info['message']}")
                new_state.processing_stats = {
                    **stats,
                    "error": f"Reranking failed: {str(e)}",
                    "total_time": time.perf_counter() - total_start,
                }
                new_state.top_passages = []
                return new_state

            # Format top_passages
            top_passages = []
            for passage in reranked[:10]:  # Ensure exactly 10 or fewer
                content = (
                    passage["content"]
                    if isinstance(passage, dict)
                    else getattr(passage, "content", "unknown")
                )
                section = (
                    passage["section"]
                    if isinstance(passage, dict)
                    else getattr(passage, "section", "unknown")
                )
                paper_id = (
                    passage["paper_id"]
                    if isinstance(passage, dict)
                    else getattr(passage, "paper_id", "unknown")
                )
                retrieval_score = (
                    passage["retrieval_score"]
                    if isinstance(passage, dict)
                    else getattr(passage, "retrieval_score", 0.0)
                )
                cross_encoder_score = (
                    passage["cross_encoder_score"]
                    if isinstance(passage, dict)
                    else getattr(passage, "cross_encoder_score", 0.0)
                )
                final_score = (
                    passage["final_score"]
                    if isinstance(passage, dict)
                    else getattr(passage, "final_score", 0.0)
                )
                top_passages.append(
                    {
                        "content": content,
                        "section": section,
                        "paper_id": paper_id,
                        "retrieval_score": retrieval_score,
                        "cross_encoder_score": cross_encoder_score,
                        "final_score": final_score,
                    }
                )

            stats["total_time"] = time.perf_counter() - total_start
            new_state.top_passages = top_passages
            new_state.processing_stats = stats

            # Export performance stats to log
            self.export_performance_log(stats)

            logger.info(
                f"ProcessingAgent: Completed in {stats['total_time']:.3f}s, "
                f"processed {stats['processed_papers']}/{stats['total_papers']} papers, "
                f"chunks {stats['chunks_embedded']}, top passages {len(top_passages)}"
            )

        except Exception as e:
            stats["total_time"] = time.perf_counter() - total_start
            stats["error"] = str(e)
            new_state.processing_stats = stats
            new_state.top_passages = []
            logger.error(f"ProcessingAgent failed: {e}")
            raise

        return new_state

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        """
        Export processing stats as a JSON-serializable dictionary.

        Args:
            minimal: If True, exclude detailed stage_times and paper_errors
                     to reduce log file size for monitoring.

        Returns:
            Dictionary suitable for JSON export
        """
        if not hasattr(self, "_last_stats") or self._last_stats is None:
            return {"error": "No processing stats available"}

        stats = self._last_stats.copy()
        stats["export_timestamp"] = time.time()
        stats["agent_version"] = "1.1.0"  # Version for tracking improvements

        if minimal:
            # Remove detailed fields for minimal export
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
        Export processing stats to performance log file.

        Args:
            stats: Processing statistics to export
            log_file: Log file path (default: performance_log.json)
        """
        try:
            # Store stats for to_dict method
            self._last_stats = stats

            export_data = self.to_dict(minimal=True)
            export_data["event"] = "processing_agent_complete"
            export_data["timestamp"] = time.time()

            # Write to log file
            import json

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(export_data) + "\n")

            logger.info(f"ProcessingAgent: Exported performance stats to {log_file}")

        except Exception as e:
            logger.warning(f"ProcessingAgent: Failed to export performance log: {e}")

    def _process_papers_to_chunks(
        self,
        papers: List[Any],
        pdf_lookup: Dict[str, PDFContent],
        stats: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process papers into chunks with section detection and fallback chunking.

        Args:
            papers: List of filtered papers
            pdf_lookup: Dict mapping paper_id to PDFContent
            stats: Processing stats to update

        Returns:
            Tuple of (all_chunks, paper_processing_stats)
        """
        all_chunks = []
        paper_stats = {"processed": 0, "skipped": 0, "errors": []}

        section_detection_total = 0.0
        chunking_total = 0.0

        for paper in papers:
            paper_id = paper.paper_id

            try:
                # Find PDF content
                if paper_id not in pdf_lookup:
                    error_info = {
                        "paper_id": paper_id,
                        "stage": "pdf_lookup",
                        "error_type": "MissingPDFContent",
                        "message": f"No PDF content found for paper {paper_id}",
                        "timestamp": time.time(),
                    }
                    stats["paper_errors"].append(error_info)
                    logger.warning(f"ProcessingAgent: {error_info['message']}")
                    paper_stats["skipped"] += 1
                    continue

                pdf_content = pdf_lookup[paper_id]

                # Section detection
                try:
                    section_start = time.perf_counter()
                    sections = self.section_detector.detect_sections(
                        pdf_content.raw_text
                    )
                    section_time = time.perf_counter() - section_start
                    section_detection_total += section_time

                    # Check if fallback needed
                    use_fallback = len(sections) < 2
                    if use_fallback:
                        stats["fallback_used_count"] += 1

                except Exception as e:
                    error_info = {
                        "paper_id": paper_id,
                        "stage": "section_detection",
                        "error_type": type(e).__name__,
                        "message": f"Section detection failed: {str(e)}",
                        "timestamp": time.time(),
                    }
                    stats["paper_errors"].append(error_info)
                    logger.warning(f"ProcessingAgent: {error_info['message']}")
                    paper_stats["skipped"] += 1
                    continue

                # Chunking
                try:
                    chunk_start = time.perf_counter()
                    if use_fallback:
                        # Create fallback chunks
                        raw_chunks = self._chunk_text_fallback(pdf_content.raw_text)
                        paper_chunks = [
                            {
                                "content": chunk,
                                "section": "unknown",
                                "paper_id": paper_id,
                                "paper_title": paper.title,
                            }
                            for chunk in raw_chunks
                        ]
                    else:
                        # Create chunks per section
                        paper_chunks = []
                        for section in sections:
                            section_chunks = self._chunk_text_by_section(
                                section["content"]
                            )
                            for chunk in section_chunks:
                                paper_chunks.append(
                                    {
                                        "content": chunk,
                                        "section": section["section"],
                                        "paper_id": paper_id,
                                        "paper_title": paper.title,
                                    }
                                )

                    chunk_time = time.perf_counter() - chunk_start
                    chunking_total += chunk_time

                except Exception as e:
                    error_info = {
                        "paper_id": paper_id,
                        "stage": "chunking",
                        "error_type": type(e).__name__,
                        "message": f"Chunking failed: {str(e)}",
                        "timestamp": time.time(),
                    }
                    stats["paper_errors"].append(error_info)
                    logger.warning(f"ProcessingAgent: {error_info['message']}")
                    paper_stats["skipped"] += 1
                    continue

                # Only add chunks if we have some
                if paper_chunks:
                    all_chunks.extend(paper_chunks)
                    stats["total_sections"] += len(sections)
                    stats["total_chunks"] += len(paper_chunks)
                    paper_stats["processed"] += 1

                    logger.info(
                        f"ProcessingAgent: Processed paper {paper_id[:8]}... "
                        f"sections: {len(sections)}, chunks: {len(paper_chunks)}, "
                        f"fallback: {use_fallback}"
                    )
                else:
                    error_info = {
                        "paper_id": paper_id,
                        "stage": "chunking",
                        "error_type": "EmptyChunks",
                        "message": f"No chunks created for paper {paper_id}",
                        "timestamp": time.time(),
                    }
                    stats["paper_errors"].append(error_info)
                    logger.warning(f"ProcessingAgent: {error_info['message']}")
                    paper_stats["skipped"] += 1

            except Exception as e:
                # Catch-all for unexpected errors
                error_info = {
                    "paper_id": paper_id,
                    "stage": "general",
                    "error_type": type(e).__name__,
                    "message": f"Unexpected error processing paper: {str(e)}",
                    "timestamp": time.time(),
                }
                stats["paper_errors"].append(error_info)
                logger.error(f"ProcessingAgent: {error_info['message']}")
                paper_stats["errors"].append({"paper_id": paper_id, "error": str(e)})
                paper_stats["skipped"] += 1
                continue

        stats["stage_times"]["section_detection"] = section_detection_total
        stats["stage_times"]["chunking"] = chunking_total
        stats["processed_papers"] = paper_stats["processed"]
        stats["skipped_papers"] = paper_stats["skipped"]

        return all_chunks, paper_stats

    def _chunk_text_fallback(
        self, text: str, chunk_size: int = 1200, overlap: int = 200
    ) -> List[str]:
        """
        Fallback chunking when section detection finds <2 sections.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to end at sentence boundary
            if end < len(text):
                last_period = text.rfind(". ", start, end)
                if last_period > end - 300:  # Don't cut too short
                    end = last_period + 2

            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 100:  # Minimum meaningful chunk
                chunks.append(chunk)

            start = max(start + chunk_size - overlap, end)

        return chunks

    def _chunk_text_by_section(
        self, section_text: str, chunk_size: int = 1000, overlap: int = 150
    ) -> List[str]:
        """
        Chunk section text into overlapping chunks.
        """
        if not section_text or len(section_text) <= chunk_size:
            return [section_text] if section_text else []

        chunks = []
        start = 0

        while start < len(section_text):
            end = start + chunk_size

            if end < len(section_text):
                last_period = section_text.rfind(". ", start, end)
                if last_period > end - 300:
                    end = last_period + 2

            chunk = section_text[start:end].strip()
            if chunk and len(chunk) >= 50:
                chunks.append(chunk)

            start = max(start + chunk_size - overlap, end)

        return chunks

    def _warmup_models(self, app: FastAPI) -> None:
        """
        Warm up and validate cached models at initialization.

        Args:
            app: FastAPI application with cached models
        """
        try:
            warmup_start = time.perf_counter()

            # Validate embedding model
            if (
                not hasattr(app.state, "embedding_model")
                or app.state.embedding_model is None
            ):
                raise RuntimeError("Embedding model not cached in app.state")

            # Quick warmup encoding
            test_embedding = app.state.embedding_model.encode(
                ["warmup test"], show_progress_bar=False
            )

            # Handle different return types (numpy array, list, etc.)
            if hasattr(test_embedding, "shape"):
                embedding_dim = test_embedding.shape[-1]  # Handle different shapes
            elif isinstance(test_embedding, list) and len(test_embedding) > 0:
                embedding_dim = (
                    len(test_embedding[0])
                    if isinstance(test_embedding[0], (list, tuple))
                    else (
                        len(test_embedding[0])
                        if hasattr(test_embedding[0], "__len__")
                        else 384
                    )
                )  # fallback
            else:
                embedding_dim = 384  # fallback for unknown types

            if embedding_dim != 384:
                raise RuntimeError(
                    f"Unexpected embedding dimensions: {embedding_dim}, expected 384"
                )

            warmup_time = time.perf_counter() - warmup_start
            logger.info(
                f"ProcessingAgent: Models warmed up successfully in {warmup_time:.3f}s, "
                f"embedding_dim: {embedding_dim}"
            )

        except Exception as e:
            logger.error(f"ProcessingAgent: Model warmup failed: {e}")
            raise RuntimeError(f"Model warmup failed: {e}") from e
