"""Retrieval module for hybrid search with LlamaIndex.

This module provides:
- LlamaIndexRetriever: Hybrid search combining FAISS + BM25 with RRF fusion
- CrossEncoderReranker: Re-ranking with cross-encoder for improved relevance
"""

from retrieval.llamaindex_retriever import LlamaIndexRetriever

# Lazy import for CrossEncoderReranker to avoid torch DLL issues on Windows
# Import it directly when needed: from retrieval.reranker import CrossEncoderReranker


def get_reranker():
    """Lazy loader for CrossEncoderReranker to avoid Windows DLL conflicts."""
    from retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker


__all__ = ["LlamaIndexRetriever", "get_reranker"]
