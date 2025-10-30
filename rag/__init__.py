"""
RAG (Retrieval-Augmented Generation) System for Necthrall

This package contains modular components for document processing in RAG pipelines:
- AdvancedDocumentChunker: Token-based section-aware text chunking with intelligent overlap
- Chunk: Pydantic model for document chunks with comprehensive metadata
- EmbeddingGenerator: Batch embedding generation with async support
- create_embedding_generator_from_app: Factory for FastAPI integration

Designed for processing scientific literature with high performance and reliability.
"""

from .chunking import AdvancedDocumentChunker
from .embeddings import EmbeddingGenerator, create_embedding_generator_from_app
from models.state import Chunk

__all__ = [
    "AdvancedDocumentChunker",
    "Chunk",
    "EmbeddingGenerator",
    "create_embedding_generator_from_app",
]
