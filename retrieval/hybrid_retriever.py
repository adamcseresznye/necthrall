from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import time
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json
import pickle

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of hybrid retrieval with fusion metadata."""

    retrieval_score: float
    doc_id: int
    fusion_method: str = "RRF"

    def __post_init__(self):
        """Validate retrieval result fields."""
        if not isinstance(self.retrieval_score, (int, float)):
            raise ValueError(
                f"retrieval_score must be numeric, got {type(self.retrieval_score)}"
            )
        if not isinstance(self.doc_id, int) or self.doc_id < 0:
            raise ValueError(f"doc_id must be non-negative int, got {self.doc_id}")
        if self.fusion_method not in ["RRF", "BM25", "semantic"]:
            raise ValueError(
                f"fusion_method must be one of ['RRF', 'BM25', 'semantic'], got {self.fusion_method}"
            )


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 keyword search with semantic similarity search.

    Uses Reciprocal Rank Fusion (RRF) with configurable k parameter to combine rankings
    from BM25 (keyword frequency) and FAISS (semantic similarity) indices.

    RRF Algorithm: score_i = (1/(rank_bm25 + k) + 1/(rank_faiss + k)) / k
    where k=60 provides balance between precision and recall for academic retrieval.

    Performance targets:
    - Index building: < 2 seconds for 10,000 document chunks
    - Query retrieval: < 500ms for top-25 results
    - Memory: < 1GB for indices and intermediate results
    - Precision@10: ≥ 0.7 on academic paper retrieval tasks
    """

    def __init__(self, rrf_k: int = 60):
        """
        Initialize hybrid retriever with RRF parameters.

        Args:
            rrf_k: Reciprocal Rank Fusion parameter (default: 60)
                  Higher values favor higher-ranked documents less.
        """
        self.rrf_k = rrf_k
        self.bm25_index: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.built = False

        # Performance tracking
        self.build_time_ms = 0.0
        self.query_time_ms = 0.0

        # Index persistence
        self.index_dir = os.path.expanduser(
            "~/.necthrall_cache"
        )  # Default cache directory
        os.makedirs(self.index_dir, exist_ok=True)

        # Query caching (LRU with embedding similarity)
        self.query_cache = {}
        self.max_cache_size = 100
        self.cache_hit_rates = {"bm25": [], "semantic": [], "rrf": []}

    def build_indices(
        self, chunks: List[Dict[str, Any]], use_cache: bool = True
    ) -> float:
        """
        Build BM25 and FAISS indices from document chunks with optional caching.

        Args:
            chunks: List of chunk dictionaries with 'content' and 'embedding' fields
            use_cache: Whether to check for cached indices first

        Returns:
            Build time in milliseconds (0.0 if loaded from cache)

        Raises:
            ValueError: If chunks are invalid or missing required fields
        """
        start_time = time.perf_counter()

        if not chunks:
            raise ValueError("Cannot build indices from empty chunk list")

        # Validate chunk structure comprehensively
        self._validate_chunks(chunks)

        # Check for cached indices if enabled
        if use_cache and self.load_indices():
            logger.info("Using cached indices instead of rebuilding")
            return 0.0  # No build time since we loaded from cache

        self.chunks = chunks

        try:
            logger.info(f"Building hybrid indices for {len(chunks)} chunks")

            # Build BM25 index with fast tokenization
            self._build_bm25_index_fast()

            # Build FAISS index
            self._build_faiss_index_fast()

            self.built = True
            build_time = (time.perf_counter() - start_time) * 1000
            self.build_time_ms = build_time

            logger.info(f"🚀 INDEX_BUILD: {len(chunks)} chunks in {build_time:.3f}ms")

            # Skip expensive statistical logging to speed up build
            # Statistics will be logged after retrieval operations

            return build_time

        except Exception as e:
            logger.error(f"Failed to build hybrid indices: {e}")
            self.built = False
            raise RuntimeError(f"Index building failed: {e}") from e

    def _log_build_statistics(
        self, chunks: List[Dict[str, Any]], build_time_ms: float
    ) -> None:
        """Log comprehensive index building statistics."""
        # Calculate tokenization statistics
        total_tokens = 0
        total_chars = 0
        doc_lengths = []

        for chunk in chunks:
            tokens = self._safe_tokenize(chunk["content"])
            total_tokens += len(tokens)
            total_chars += len(chunk["content"])
            doc_lengths.append(len(tokens))

        avg_doc_length = np.mean(doc_lengths) if doc_lengths else 0
        std_doc_length = np.std(doc_lengths) if doc_lengths else 0
        vocab_size = len(
            set(
                token
                for chunk in chunks
                for token in self._safe_tokenize(chunk["content"])
            )
        )

        # Memory breakdown
        bm25_memory = total_tokens * 8 / (1024 * 1024) if self.bm25_index else 0
        faiss_memory = len(chunks) * 384 * 4 / (1024 * 1024) if self.faiss_index else 0
        content_memory = sum(len(chunk["content"]) * 2 for chunk in chunks) / (
            1024 * 1024
        )

        logger.info(
            json.dumps(
                {
                    "event": "hybrid_index_build_detailed",
                    "chunks_processed": len(chunks),
                    "total_tokens": total_tokens,
                    "total_chars": total_chars,
                    "vocab_size": vocab_size,
                    "avg_doc_tokens": round(avg_doc_length, 1),
                    "std_doc_tokens": round(std_doc_length, 1),
                    "bm25_index_memory_mb": round(bm25_memory, 2),
                    "faiss_index_memory_mb": round(faiss_memory, 2),
                    "content_memory_mb": round(content_memory, 2),
                    "total_memory_mb": round(
                        bm25_memory + faiss_memory + content_memory, 2
                    ),
                    "build_time_ms": round(build_time_ms, 2),
                    "throughput_chunks_per_sec": round(
                        len(chunks) / (build_time_ms / 1000), 2
                    ),
                    "rrf_k": self.rrf_k,
                }
            )
        )

    def _build_bm25_index_safe(self) -> None:
        """Build BM25 index with enhanced error handling and tokenization."""
        corpus = []

        for i, chunk in enumerate(self.chunks):
            try:
                # Safe tokenization with fallback
                tokens = self._safe_tokenize(chunk["content"])
                corpus.append(tokens)
            except Exception as e:
                logger.warning(f"Failed to tokenize chunk {i}: {e}")
                # Add placeholder tokens to maintain alignment
                corpus.append(["[tokenization_error]"])

        try:
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"BM25 index built with {len(corpus)} documents")
        except Exception as e:
            logger.error(f"BM25 index construction failed: {e}")
            raise RuntimeError(f"BM25 index construction failed: {e}") from e

    def retrieve(
        self, query: str, query_embedding: np.ndarray, top_k: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid BM25 + semantic search with RRF fusion.

        Args:
            query: Search query string
            query_embedding: Query embedding vector (shape: (384,))
            top_k: Number of top results to return (default: 25)

        Returns:
            List of enriched chunk dictionaries with retrieval metadata

        Raises:
            RuntimeError: If indices not built or query validation fails
            ValueError: If query_embedding has wrong dimensions
        """
        if not self.built:
            raise RuntimeError(
                "Retrieval indices not built. Call build_indices() first."
            )

        query_start = time.perf_counter()
        logger.info(f"Hybrid retrieval for query: '{query[:50]}...' (top_k={top_k})")

        # Validate inputs
        if not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []

        if query_embedding.shape != (384,):
            raise ValueError(
                f"query_embedding must be shape (384,), got {query_embedding.shape}"
            )

        try:
            # Get BM25 scores for all documents
            bm25_scores = self._get_bm25_scores(query)

            # Get FAISS similarity scores for all documents
            faiss_similarities = self._get_faiss_similarities(query_embedding)

            # Apply RRF fusion
            fused_results = self._apply_rrf_fusion(
                bm25_scores, faiss_similarities, top_k
            )

            # Enrich chunks with retrieval metadata
            results = []
            for result in fused_results:
                chunk_copy = self.chunks[result.doc_id].copy()
                chunk_copy["retrieval_score"] = result.retrieval_score
                chunk_copy["doc_id"] = result.doc_id
                chunk_copy["fusion_method"] = result.fusion_method
                results.append(chunk_copy)

            query_time = (time.perf_counter() - query_start) * 1000
            self.query_time_ms = query_time

            logger.info(
                json.dumps(
                    {
                        "event": "hybrid_retrieval_complete",
                        "query_length": len(query),
                        "results_count": len(results),
                        "top_score": results[0]["retrieval_score"] if results else 0.0,
                        "query_time_ms": round(query_time, 2),
                        "fusion_method": "RRF",
                        "rrf_k": self.rrf_k,
                    }
                )
            )

            return results

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise RuntimeError(f"Query retrieval failed: {e}") from e

    def _validate_chunks_minimal(self, chunks: List[Dict[str, Any]]) -> bool:
        """Minimal validation for fast build - check first chunk only."""
        if not isinstance(chunks, list) or not chunks:
            return False

        # Spot check first chunk only for speed
        chunk = chunks[0]
        if not isinstance(chunk, dict):
            return False
        if "content" not in chunk or "embedding" not in chunk:
            return False
        if not isinstance(chunk["content"], str):
            return False

        embedding = chunk["embedding"]
        if not isinstance(embedding, np.ndarray) or embedding.shape != (384,):
            return False

        return True

    def _validate_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Comprehensive validation of all chunks.

        Args:
            chunks: List of chunk dictionaries to validate

        Raises:
            ValueError: If any chunk is invalid
        """
        if not chunks:
            raise ValueError("Cannot validate empty chunk list")

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {i} must be a dictionary, got {type(chunk)}")

            # Check required fields
            if "content" not in chunk:
                raise ValueError(f"Chunk {i} missing required 'content' field")

            content = chunk["content"]
            if not isinstance(content, str):
                raise ValueError(
                    f"Chunk {i} content must be string, got {type(content)}"
                )
            if not content.strip():
                raise ValueError(f"Chunk {i} has empty or non-string content")

            if "embedding" not in chunk:
                raise ValueError(f"Chunk {i} missing required 'embedding' field")

            embedding = chunk["embedding"]
            if not isinstance(embedding, np.ndarray):
                raise ValueError(
                    f"Chunk {i} embedding must be numpy array, got {type(embedding)}"
                )
            if embedding.shape != (384,):
                raise ValueError(
                    f"Chunk {i} embedding shape must be (384,), got {embedding.shape}"
                )
            if embedding.dtype not in [np.float32, np.float64]:
                raise ValueError(
                    f"Chunk {i} embedding must be float type, got {embedding.dtype}"
                )

    def _build_bm25_index_fast(self) -> None:
        """Build BM25 index with vectorized tokenization for speed."""
        # Vectorized tokenization: much faster than per-chunk loop
        corpus = [chunk["content"].lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(corpus)} documents")

    def _build_faiss_index_fast(self) -> None:
        """Build FAISS semantic similarity index optimized for speed."""
        # Fast numpy stacking for embeddings
        embeddings_matrix = np.array(
            [chunk["embedding"] for chunk in self.chunks]
        ).astype(np.float32)

        # Create FAISS index (IndexFlatIP for inner product/cosine similarity)
        dim = 384
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings_matrix)

        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def _get_bm25_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for query against all documents."""
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        return np.array(scores)

    def _get_faiss_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Get semantic similarities for query embedding against all documents."""
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS index (k = all documents for full ranking)
        k = len(self.chunks)
        similarities, indices = self.faiss_index.search(
            query_norm.reshape(1, -1).astype(np.float32), k
        )

        return similarities[0]  # FAISS returns shape (1, k), we want (k,)

    def _apply_rrf_fusion(
        self, bm25_scores: np.ndarray, faiss_similarities: np.ndarray, top_k: int
    ) -> List[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion to combine BM25 and semantic rankings.

        Args:
            bm25_scores: BM25 scores for all documents
            faiss_similarities: Semantic similarities for all documents
            top_k: Number of top results to return

        Returns:
            List of RetrievalResult objects sorted by RRF score (descending)
        """
        # Get rankings (1-based, higher score = better rank)
        bm25_ranks = self._scores_to_ranks(bm25_scores)
        faiss_ranks = self._scores_to_ranks(faiss_similarities)

        # Apply RRF formula
        rrf_scores = []
        for doc_id in range(len(self.chunks)):
            bm25_contrib = 1.0 / (bm25_ranks[doc_id] + self.rrf_k)
            faiss_contrib = 1.0 / (faiss_ranks[doc_id] + self.rrf_k)
            rrf_score = (bm25_contrib + faiss_contrib) / self.rrf_k
            rrf_scores.append(rrf_score)

        # Sort by RRF score and get top_k
        sorted_indices = np.argsort(rrf_scores)[::-1][:top_k]
        results = [
            RetrievalResult(
                retrieval_score=rrf_scores[idx], doc_id=int(idx), fusion_method="RRF"
            )
            for idx in sorted_indices
        ]

        return results

    def _scores_to_ranks(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert scores to 1-based ranks (higher score = lower rank number).

        Args:
            scores: Array of scores

        Returns:
            Array of 1-based ranks
        """
        # Get rank positions (0-based, then convert to 1-based)
        # argsort gives ascending order, so we reverse it for descending
        ranks = np.argsort(scores)[::-1]
        rank_dict = {idx: rank + 1 for rank, idx in enumerate(ranks)}
        return np.array([rank_dict[i] for i in range(len(scores))])

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of indices in MB."""
        memory_mb = 0.0

        # BM25 memory (rough estimate: token count * 8 bytes)
        if self.bm25_index:
            # Estimate based on average document length and corpus size
            avg_doc_len = (
                sum(self.bm25_index.doc_len) / len(self.bm25_index.doc_len)
                if self.bm25_index.doc_len
                else 10
            )
            total_tokens = int(avg_doc_len * self.bm25_index.corpus_size)
            memory_mb += total_tokens * 8 / (1024 * 1024)

        # FAISS memory (vectors * dimensions * 4 bytes per float32)
        if self.faiss_index:
            memory_mb += self.faiss_index.ntotal * 384 * 4 / (1024 * 1024)

        # Chunk content (rough: len(content) * 2 bytes per char)
        content_memory = sum(len(chunk["content"]) * 2 for chunk in self.chunks) / (
            1024 * 1024
        )
        memory_mb += content_memory

        return memory_mb

    def save_indices(self, index_path: Optional[str] = None) -> str:
        """
        Save indices to disk for persistence.

        Args:
            index_path: Optional path to save indices. If None, auto-generates based on content hash.

        Returns:
            Path where indices were saved

        Raises:
            RuntimeError: If indices not built or save fails
        """
        if not self.built:
            raise RuntimeError("Cannot save indices that haven't been built")

        try:
            if index_path is None:
                # Generate hash based on chunk content for cache key
                content_hash = self._compute_content_hash()
                index_path = os.path.join(self.index_dir, f"indices_{content_hash}.pkl")

            os.makedirs(os.path.dirname(index_path), exist_ok=True)

            # Save indices and metadata
            index_data = {
                "bm25_index": self.bm25_index,
                "faiss_index": self.faiss_index,
                "chunks": self.chunks,
                "rrf_k": self.rrf_k,
                "build_time_ms": self.build_time_ms,
                "metadata": {
                    "created_at": time.time(),
                    "chunk_count": len(self.chunks),
                    "content_hash": self._compute_content_hash(),
                },
            }

            with open(index_path, "wb") as f:
                pickle.dump(index_data, f)

            logger.info(f"Indices saved to {index_path}")
            return index_path

        except Exception as e:
            logger.error(f"Failed to save indices: {e}")
            raise RuntimeError(f"Index saving failed: {e}") from e

    def load_indices(self, index_path: Optional[str] = None) -> bool:
        """
        Load indices from disk.

        Args:
            index_path: Path to load indices from. If None, tries to find based on content hash.

        Returns:
            True if indices were loaded successfully, False otherwise
        """
        try:
            if index_path is None:
                # Try to find cached indices based on content hash
                if not self.chunks:
                    logger.warning("No chunks available for hash-based loading")
                    return False

                content_hash = self._compute_content_hash()
                index_path = os.path.join(self.index_dir, f"indices_{content_hash}.pkl")

            if not os.path.exists(index_path):
                logger.info(f"No cached indices found at {index_path}")
                return False

            with open(index_path, "rb") as f:
                index_data = pickle.load(f)

            # Validate loaded data
            if not self._validate_loaded_indices(index_data):
                logger.warning("Loaded index data validation failed")
                return False

            # Load indices
            self.bm25_index = index_data["bm25_index"]
            self.faiss_index = index_data["faiss_index"]
            self.chunks = index_data["chunks"]
            self.rrf_k = index_data.get("rrf_k", 60)
            self.build_time_ms = index_data.get("build_time_ms", 0.0)
            self.built = True

            logger.info(
                json.dumps(
                    {
                        "event": "indices_loaded_from_cache",
                        "path": index_path,
                        "chunks_loaded": len(self.chunks),
                        "cache_age_seconds": time.time()
                        - index_data.get("metadata", {}).get("created_at", time.time()),
                    }
                )
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to load indices: {e}")
            return False

    def _compute_content_hash(self) -> str:
        """Compute hash of chunk content for cache keying."""
        if not self.chunks:
            return "empty"

        # Hash content of first, middle, and last chunks for representative sample
        content_parts = []
        if self.chunks:
            content_parts.append(self.chunks[0]["content"])
            if len(self.chunks) > 2:
                content_parts.append(self.chunks[len(self.chunks) // 2]["content"])
            if len(self.chunks) > 1:
                content_parts.append(self.chunks[-1]["content"])

        content_str = "|".join(content_parts) + f"|{len(self.chunks)}"
        return hashlib.md5(content_str.encode()).hexdigest()[:16]

    def _validate_loaded_indices(self, index_data: Dict[str, Any]) -> bool:
        """Validate loaded index data integrity."""
        try:
            # Check required keys
            required_keys = ["bm25_index", "faiss_index", "chunks", "rrf_k"]
            if not all(key in index_data for key in required_keys):
                return False

            # Validate BM25 index
            bm25 = index_data["bm25_index"]
            if not hasattr(bm25, "corpus_size") or bm25.corpus_size != len(
                index_data["chunks"]
            ):
                return False

            # Validate FAISS index
            faiss_idx = index_data["faiss_index"]
            if not hasattr(faiss_idx, "ntotal") or faiss_idx.ntotal != len(
                index_data["chunks"]
            ):
                return False

            # Validate chunk structure (spot check first chunk)
            chunks = index_data["chunks"]
            if chunks and not self._spot_check_chunks(chunks):
                return False

            return True

        except Exception as e:
            logger.warning(f"Index validation failed: {e}")
            return False

    def _spot_check_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Spot check chunk validity."""
        if not chunks:
            return False

        # Check first chunk
        chunk = chunks[0]
        if not isinstance(chunk, dict):
            return False
        if "content" not in chunk or "embedding" not in chunk:
            return False
        if not isinstance(chunk["content"], str):
            return False

        embedding = chunk["embedding"]
        if not isinstance(embedding, np.ndarray) or embedding.shape != (384,):
            return False

        return True

    def _check_faiss_integrity(self) -> bool:
        """Check FAISS index integrity and rebuild if corrupted."""
        if not self.faiss_index:
            return False

        try:
            # Try a simple search operation to test integrity
            test_query = np.ones(384, dtype=np.float32)
            test_query = test_query / np.linalg.norm(test_query)

            _, indices = self.faiss_index.search(
                test_query.reshape(1, -1).astype(np.float32),
                min(5, self.faiss_index.ntotal),
            )

            # Check if indices are valid
            if np.any(indices < 0) or np.any(indices >= self.faiss_index.ntotal):
                logger.warning("FAISS index returned invalid indices, may be corrupted")
                return False

            return True

        except Exception as e:
            logger.warning(f"FAISS integrity check failed: {e}")
            return False

    def _safe_tokenize(self, text: str) -> List[str]:
        """Safely tokenize text with fallback handling."""
        if not isinstance(text, str):
            logger.warning(f"Non-string input to tokenizer: {type(text)}")
            text = str(text)

        try:
            # Primary tokenization
            tokens = text.lower().split()
            return tokens
        except Exception as e:
            logger.warning(f"Primary tokenization failed: {e}")
            try:
                # Fallback: simple character-based tokenization
                tokens = list(text.lower())
                return tokens
            except Exception as fallback_e:
                logger.error(f"Fallback tokenization also failed: {fallback_e}")
                return []

    def retrieve_with_enhanced_logging(
        self, query: str, query_embedding: np.ndarray, top_k: int = 25
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve documents with enhanced logging and performance metrics.

        Returns:
            Tuple of (results, metrics_dict)
        """
        if not self.built:
            raise RuntimeError(
                "Retrieval indices not built. Call build_indices() first."
            )

        start_time = time.perf_counter()

        # Validate inputs
        if not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return [], {"status": "empty_query"}

        if query_embedding.shape != (384,):
            raise ValueError(
                f"query_embedding must be shape (384,), got {query_embedding.shape}"
            )

        # Check cache first
        cache_key = self._get_cache_key(query, query_embedding, top_k)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Cache hit - return cached result with metrics
            self.cache_hit_rates["rrf"].append(1.0)  # Cache hit
            logger.info("Using cached retrieval result")

            # Provide basic cache hit metrics
            cache_metrics = {
                "status": "cached",
                "query_length": len(query),
                "top_k": top_k,
                "query_time_ms": 0.0,  # No computation time for cached result
                "cache_hit_rate_rrf": round(
                    (
                        np.mean(self.cache_hit_rates["rrf"][-10:])
                        if self.cache_hit_rates["rrf"]
                        else 0.0
                    ),
                    3,
                ),
                "cache_size": len(self.query_cache),
            }
            return cached_result, cache_metrics

        try:
            # Get individual method scores for detailed analysis
            bm25_scores = self._get_bm25_scores(query)
            faiss_similarities = self._get_faiss_similarities(query_embedding)

            # Compute rankings and fusion
            bm25_ranks = self._scores_to_ranks(bm25_scores)
            faiss_ranks = self._scores_to_ranks(faiss_similarities)
            rrf_results = self._apply_rrf_fusion(bm25_scores, faiss_similarities, top_k)

            # Build results
            results = []
            for result in rrf_results:
                chunk_copy = self.chunks[result.doc_id].copy()
                chunk_copy["retrieval_score"] = result.retrieval_score
                chunk_copy["doc_id"] = result.doc_id
                chunk_copy["fusion_method"] = result.fusion_method
                results.append(chunk_copy)

            # Cache the result
            self._cache_result(cache_key, results)

            # Compute detailed metrics
            metrics = self._compute_detailed_metrics(
                query,
                bm25_scores,
                bm25_ranks,
                faiss_similarities,
                faiss_ranks,
                rrf_results,
                start_time,
                top_k,
            )

            # Log performance analysis
            logger.info(json.dumps(metrics))

            # Update hit rate tracking (cache miss)
            self.cache_hit_rates["rrf"].append(0.0)

            return results, metrics

        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            raise RuntimeError(f"Query retrieval failed: {e}") from e

    def _get_cache_key(
        self, query: str, query_embedding: np.ndarray, top_k: int
    ) -> str:
        """Generate cache key for query."""
        # Use query + quantized embedding + top_k
        emb_quantized = (query_embedding * 1000).astype(int)  # Quantize for similarity
        emb_str = hashlib.md5(np.sort(emb_quantized).tobytes()).hexdigest()[:8]
        return f"{query.strip()}_{emb_str}_{top_k}"

    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Check cache for result."""
        return self.query_cache.get(cache_key)

    def _cache_result(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Cache retrieval result with LRU eviction."""
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = results

    def _compute_detailed_metrics(
        self,
        query: str,
        bm25_scores: np.ndarray,
        bm25_ranks: np.ndarray,
        faiss_similarities: np.ndarray,
        faiss_ranks: np.ndarray,
        rrf_results: List[RetrievalResult],
        start_time: float,
        top_k: int,
    ) -> Dict[str, Any]:
        """Compute detailed performance and quality metrics."""

        query_time_ms = (time.perf_counter() - start_time) * 1000

        # Top-k analysis
        bm25_top_k = np.argsort(bm25_scores)[-top_k:][::-1]
        faiss_top_k = np.argsort(faiss_similarities)[-top_k:][::-1]
        rrf_top_k = [r.doc_id for r in rrf_results]

        # Hit rate analysis
        bm25_precision = len(set(bm25_top_k) & set(rrf_top_k)) / max(len(rrf_top_k), 1)
        faiss_precision = len(set(faiss_top_k) & set(rrf_top_k)) / max(
            len(rrf_top_k), 1
        )

        # Score statistics
        bm25_avg_top_k = (
            float(np.mean(bm25_scores[bm25_top_k])) if bm25_top_k.size > 0 else 0.0
        )
        faiss_avg_top_k = (
            float(np.mean(faiss_similarities[faiss_top_k]))
            if faiss_top_k.size > 0
            else 0.0
        )
        rrf_avg_score = (
            float(np.mean([r.retrieval_score for r in rrf_results]))
            if rrf_results
            else 0.0
        )

        # Diversity metrics
        bm25_faiss_overlap = len(set(bm25_top_k) & set(faiss_top_k)) / max(top_k, 1)

        return {
            "event": "hybrid_retrieval_detailed",
            "query_length": len(query),
            "top_k": top_k,
            "query_time_ms": round(query_time_ms, 2),
            "bm25_avg_score_top_k": round(bm25_avg_top_k, 4),
            "faiss_avg_similarity_top_k": round(faiss_avg_top_k, 4),
            "rrf_avg_score": round(rrf_avg_score, 4),
            "bm25_precision_in_rrf": round(bm25_precision, 3),
            "faiss_precision_in_rrf": round(faiss_precision, 3),
            "bm25_faiss_overlap": round(bm25_faiss_overlap, 3),
            "cache_size": len(self.query_cache),
            "cache_hit_rate_rrf": round(
                (
                    np.mean(self.cache_hit_rates["rrf"][-10:])
                    if self.cache_hit_rates["rrf"]
                    else 0.0
                ),
                3,
            ),  # Last 10 queries
            "rrf_k": self.rrf_k,
            "fusion_improvement": self._calculate_fusion_improvement(
                bm25_scores, faiss_similarities, rrf_results
            ),
        }

    def _calculate_fusion_improvement(
        self,
        bm25_scores: np.ndarray,
        faiss_scores: np.ndarray,
        rrf_results: List[RetrievalResult],
    ) -> Dict[str, float]:
        """Calculate various metrics of fusion improvement."""
        if not rrf_results:
            return {"score_variance": 0.0, "method_diversity": 0.0}

        # Score variance analysis (more even distribution = better fusion)
        rrf_scores = np.array([r.retrieval_score for r in rrf_results])
        score_variance = float(np.var(rrf_scores)) if len(rrf_scores) > 1 else 0.0

        # Method diversity (how different the top results are from individual methods)
        rrf_ids = {r.doc_id for r in rrf_results}
        bm25_top = set(np.argsort(bm25_scores)[-len(rrf_results) :][::-1])
        faiss_top = set(np.argsort(faiss_scores)[-len(rrf_results) :][::-1])

        rrf_from_bm25 = len(rrf_ids & bm25_top) / max(len(rrf_ids), 1)
        rrf_from_faiss = len(rrf_ids & faiss_top) / max(len(rrf_ids), 1)
        method_diversity = (rrf_from_bm25 + rrf_from_faiss) / 2.0

        return {
            "score_variance": round(score_variance, 4),
            "method_diversity": round(method_diversity, 3),
        }
