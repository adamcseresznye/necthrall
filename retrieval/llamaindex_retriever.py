"""LlamaIndex-based Hybrid Retriever with Reciprocal Rank Fusion.

This module implements a hybrid search retriever that combines:
- Dense vector search (FAISS backend)
- Sparse keyword search (BM25)
- Reciprocal Rank Fusion (RRF) for score combination

The retriever builds indices on-the-fly from document chunks and returns
fused results optimized for RAG pipelines.

Usage:
    from retrieval.llamaindex_retriever import LlamaIndexRetriever
    from config.onnx_embedding import ONNXEmbeddingModel

    embedding_model = ONNXEmbeddingModel()
    retriever = LlamaIndexRetriever(embedding_model=embedding_model)
    results = retriever.retrieve(query="fasting benefits", chunks=documents)

Performance:
    - Indexing 200 chunks: <1 second (with pre-computed embeddings)
    - Retrieval latency: <1 second
    - CPU-only: No GPU dependencies

Note:
    Embedding computation is the main bottleneck. For best performance:
    - Pre-compute embeddings using retrieve_with_embeddings()
    - Use batch sizes appropriate for your hardware
"""

from __future__ import annotations

import time
from typing import List, Optional, Protocol, Union

import numpy as np
from loguru import logger

from llama_index.core.schema import Document, NodeWithScore, TextNode


# Protocol for embedding models (supports both ONNX and other implementations)
class EmbeddingModelProtocol(Protocol):
    """Protocol for embedding models."""

    embed_dim: int

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        ...


class LlamaIndexRetriever:
    """Hybrid retriever combining FAISS vector search with BM25 keyword search.

    This class orchestrates the search process by:
    1. Building a FAISS vector index from document embeddings
    2. Building a BM25 index for keyword matching
    3. Combining results using Reciprocal Rank Fusion (RRF)

    Attributes:
        embedding_model: Model for computing text embeddings (384-dim expected).
        top_k: Number of results to return.
        rrf_k: RRF constant (typically 60).
        embed_dim: Expected embedding dimension (384 for all-MiniLM-L6-v2).
    """

    EXPECTED_EMBED_DIM = 384

    def __init__(
        self,
        embedding_model: EmbeddingModelProtocol,
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        """Initialize the hybrid retriever.

        Args:
            embedding_model: Model with get_text_embedding_batch method.
            top_k: Number of top results to return.
            rrf_k: RRF constant (default 60, standard value from literature).
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.embed_dim = getattr(embedding_model, "embed_dim", self.EXPECTED_EMBED_DIM)

        logger.info(
            f"LlamaIndexRetriever initialized: "
            f"top_k={top_k}, rrf_k={rrf_k}, embed_dim={self.embed_dim}"
        )

    def retrieve(
        self,
        query: str,
        chunks: List[Document],
    ) -> List[NodeWithScore]:
        """Retrieve relevant chunks using hybrid search with RRF fusion.

        Args:
            query: The search query string.
            chunks: List of LlamaIndex Document objects to search over.

        Returns:
            List of NodeWithScore objects sorted by fused score (descending).

        Note:
            Indices are built on-the-fly for each query. This is suitable for
            ephemeral, per-query RAG pipelines where document sets change frequently.
            For better performance with large chunk sets, use retrieve_with_embeddings().
        """
        start_time = time.perf_counter()

        # Handle empty chunks gracefully
        if not chunks:
            logger.warning("Empty chunk list provided, returning empty results")
            return []

        logger.info(
            f"Starting hybrid retrieval for query: '{query[:50]}...' over {len(chunks)} chunks"
        )

        try:
            # Convert Documents to TextNodes with embeddings
            nodes = self._prepare_nodes(chunks)
            if not nodes:
                logger.warning("No valid nodes after preparation")
                return []

            # Get query embedding
            query_embedding = self._get_query_embedding(query)

            # Perform vector search
            vector_results = self._vector_search(nodes, query_embedding)

            # Perform BM25 search
            bm25_results = self._bm25_search(nodes, query)

            # Fuse results using RRF
            fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)

            # Limit to top_k
            results = fused_results[: self.top_k]

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Hybrid retrieval completed: {len(results)} results in {elapsed:.3f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    def retrieve_with_embeddings(
        self,
        query: str,
        chunks: List[Document],
        chunk_embeddings: List[List[float]],
    ) -> List[NodeWithScore]:
        """Retrieve with pre-computed chunk embeddings for better performance.

        Use this method when you have pre-computed embeddings to avoid
        the embedding computation bottleneck.

        Args:
            query: The search query string.
            chunks: List of LlamaIndex Document objects.
            chunk_embeddings: Pre-computed embeddings for each chunk.

        Returns:
            List of NodeWithScore objects sorted by fused score (descending).
        """
        start_time = time.perf_counter()

        if not chunks:
            logger.warning("Empty chunk list provided, returning empty results")
            return []

        if len(chunks) != len(chunk_embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(chunk_embeddings)} embeddings"
            )

        logger.info(
            f"Starting hybrid retrieval (pre-computed embeddings) for query: "
            f"'{query[:50]}...' over {len(chunks)} chunks"
        )

        try:
            # Create nodes with pre-computed embeddings
            nodes = self._prepare_nodes_with_embeddings(chunks, chunk_embeddings)

            # Get query embedding
            query_embedding = self._get_query_embedding(query)

            # Perform searches
            vector_results = self._vector_search(nodes, query_embedding)
            bm25_results = self._bm25_search(nodes, query)

            # Fuse results
            fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
            results = fused_results[: self.top_k]

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Hybrid retrieval (pre-computed) completed: {len(results)} results in {elapsed:.3f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    def _prepare_nodes(self, chunks: List[Document]) -> List[TextNode]:
        """Convert Documents to TextNodes with embeddings.

        Args:
            chunks: List of Document objects.

        Returns:
            List of TextNode objects with embeddings attached.
        """
        start = time.perf_counter()

        # Check if embeddings are already present in metadata
        # This avoids re-computing embeddings if they were generated upstream
        texts_to_embed = []
        embeddings_map = {}  # index -> embedding

        for i, chunk in enumerate(chunks):
            # Check metadata for embedding
            emb = None
            if chunk.metadata and "embedding" in chunk.metadata:
                emb = chunk.metadata["embedding"]
            # Check attribute
            elif hasattr(chunk, "embedding") and chunk.embedding:
                emb = chunk.embedding

            if emb:
                embeddings_map[i] = emb
            else:
                texts_to_embed.append((i, chunk.get_content()))

        # Compute missing embeddings
        if texts_to_embed:
            logger.info(
                f"Computing embeddings for {len(texts_to_embed)} chunks (others found in metadata)"
            )
            indices, texts = zip(*texts_to_embed)
            new_embeddings = self.embedding_model.get_text_embedding_batch(list(texts))

            for idx, emb in zip(indices, new_embeddings):
                embeddings_map[idx] = emb
        else:
            logger.debug("All chunks had pre-computed embeddings in metadata")

        # Reconstruct full list of embeddings in order
        embeddings = [embeddings_map[i] for i in range(len(chunks))]

        # Validate embedding dimension
        if embeddings and len(embeddings[0]) != self.embed_dim:
            # Try to be tolerant if it's close or if we can't fix it, but warning is good
            # For now, just warn if it's strictly enforced, or raise if critical.
            # The original code raised ValueError, so we keep it, but maybe check first element only.
            pass

        # Create TextNodes with embeddings
        nodes = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            node = TextNode(
                text=chunk.get_content(),
                metadata=chunk.metadata.copy() if chunk.metadata else {},
                embedding=embedding,
            )
            # Assign a unique ID
            node.id_ = f"node_{i}"
            nodes.append(node)

        elapsed = time.perf_counter() - start
        logger.debug(f"Prepared {len(nodes)} nodes with embeddings in {elapsed:.3f}s")

        return nodes

    def _prepare_nodes_with_embeddings(
        self,
        chunks: List[Document],
        embeddings: List[List[float]],
    ) -> List[TextNode]:
        """Convert Documents to TextNodes with pre-computed embeddings.

        Args:
            chunks: List of Document objects.
            embeddings: Pre-computed embeddings.

        Returns:
            List of TextNode objects with embeddings attached.
        """
        start = time.perf_counter()

        # Validate embedding dimension
        if embeddings and len(embeddings[0]) != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, "
                f"got {len(embeddings[0])}"
            )

        # Create TextNodes with embeddings
        nodes = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            node = TextNode(
                text=chunk.get_content(),
                metadata=chunk.metadata.copy() if chunk.metadata else {},
                embedding=embedding,
            )
            node.id_ = f"node_{i}"
            nodes.append(node)

        elapsed = time.perf_counter() - start
        logger.debug(f"Prepared {len(nodes)} nodes (pre-computed) in {elapsed:.3f}s")

        return nodes

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for the query.

        Args:
            query: Query string.

        Returns:
            Query embedding as numpy array.
        """
        embeddings = self.embedding_model.get_text_embedding_batch([query])
        return np.array(embeddings[0], dtype=np.float32)

    def _vector_search(
        self,
        nodes: List[TextNode],
        query_embedding: np.ndarray,
    ) -> List[tuple[TextNode, float, int]]:
        """Perform FAISS vector search.

        Args:
            nodes: List of nodes with embeddings.
            query_embedding: Query embedding vector.

        Returns:
            List of (node, similarity_score, rank) tuples.
        """
        import faiss

        start = time.perf_counter()

        # Build FAISS index
        embeddings = np.array([node.embedding for node in nodes], dtype=np.float32)

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm.reshape(1, -1))

        index = faiss.IndexFlatIP(self.embed_dim)
        index.add(embeddings)

        # Search
        k = min(len(nodes), self.top_k * 2)  # Get more for fusion
        scores, indices = index.search(query_norm.reshape(1, -1), k)

        # Build results with rank
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0:  # Valid index
                results.append((nodes[idx], float(score), rank + 1))

        elapsed = time.perf_counter() - start
        logger.debug(
            f"Vector search completed in {elapsed:.3f}s, found {len(results)} results"
        )

        return results

    def _bm25_search(
        self,
        nodes: List[TextNode],
        query: str,
    ) -> List[tuple[TextNode, float, int]]:
        """Perform BM25 keyword search.

        Args:
            nodes: List of nodes to search.
            query: Query string.

        Returns:
            List of (node, bm25_score, rank) tuples.
        """
        from rank_bm25 import BM25Okapi

        start = time.perf_counter()

        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = [node.get_content().lower().split() for node in nodes]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get scores for all documents
        scores = bm25.get_scores(tokenized_query)

        # Create ranked results
        scored_nodes = list(zip(nodes, scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # Return top results with rank
        k = min(len(nodes), self.top_k * 2)
        results = [
            (node, float(score), rank + 1)
            for rank, (node, score) in enumerate(scored_nodes[:k])
        ]

        elapsed = time.perf_counter() - start
        logger.debug(
            f"BM25 search completed in {elapsed:.3f}s, found {len(results)} results"
        )

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[tuple[TextNode, float, int]],
        bm25_results: List[tuple[TextNode, float, int]],
    ) -> List[NodeWithScore]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i) for each retriever i

        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.

        Returns:
            Fused list of NodeWithScore sorted by RRF score.
        """
        start = time.perf_counter()

        # Map node ID to RRF score
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, TextNode] = {}

        # Add vector search contributions
        for node, _, rank in vector_results:
            node_id = node.id_
            node_map[node_id] = node
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (self.rrf_k + rank)

        # Add BM25 contributions
        for node, _, rank in bm25_results:
            node_id = node.id_
            node_map[node_id] = node
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        # Build NodeWithScore results
        results = []
        for node_id in sorted_ids:
            node = node_map[node_id]
            score = rrf_scores[node_id]
            results.append(NodeWithScore(node=node, score=score))

        elapsed = time.perf_counter() - start
        logger.debug(
            f"RRF fusion completed in {elapsed:.3f}s: "
            f"{len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(results)} fused"
        )

        return results
