"""Unit tests for LlamaIndexRetriever.

Tests cover:
- Basic retrieval with keyword matching
- Empty document list handling
- Integration with ONNX embedding model (mocked)
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from llama_index.core.schema import Document, NodeWithScore, TextNode


@pytest.fixture
def sample_documents():
    """Sample documents for testing retrieval."""
    return [
        Document(text="fasting is good for metabolism", metadata={"source": "doc1"}),
        Document(text="sugar is bad for health", metadata={"source": "doc2"}),
        Document(
            text="exercise improves cardiovascular health", metadata={"source": "doc3"}
        ),
        Document(
            text="intermittent fasting promotes autophagy", metadata={"source": "doc4"}
        ),
        Document(
            text="protein synthesis requires amino acids", metadata={"source": "doc5"}
        ),
    ]


@pytest.fixture
def mock_embedding_model():
    """Mock ONNX embedding model that returns 384-dim vectors."""
    mock = MagicMock()
    mock.embed_dim = 384

    def fake_embed(texts):
        """Generate deterministic embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            emb = np.random.rand(384).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
        return embeddings

    mock.get_text_embedding_batch = fake_embed
    return mock


@pytest.mark.unit
class TestLlamaIndexRetriever:
    """Test suite for LlamaIndexRetriever."""

    def test_basic_retrieval_returns_relevant_chunks(
        self, sample_documents, mock_embedding_model
    ):
        """Test that retrieval returns relevant chunks for a query."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=sample_documents)

        # Should return results
        assert len(results) > 0
        # All results should be NodeWithScore
        assert all(isinstance(r, NodeWithScore) for r in results)
        # Results should have scores
        assert all(r.score is not None for r in results)

    def test_empty_document_list_returns_empty(self, mock_embedding_model):
        """Test that empty chunk list returns empty results."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=[])

        assert results == []

    def test_retrieval_with_matching_keyword(
        self, sample_documents, mock_embedding_model
    ):
        """Test that keyword-matching chunks score higher."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=sample_documents)

        # Get text content from top results
        top_texts = [r.node.get_content() for r in results[:5]]

        # At least one should contain "fasting"
        assert any("fasting" in text.lower() for text in top_texts)

    def test_retrieval_respects_top_k(self, sample_documents, mock_embedding_model):
        """Test that retrieval returns at most top_k results."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model, top_k=2)
        results = retriever.retrieve(query="health", chunks=sample_documents)

        assert len(results) <= 2

    def test_results_are_sorted_by_score(self, sample_documents, mock_embedding_model):
        """Test that results are sorted by score descending."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=sample_documents)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_retrieval_preserves_metadata(self, sample_documents, mock_embedding_model):
        """Test that document metadata is preserved in results."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=sample_documents)

        # Check that metadata is preserved
        for result in results:
            assert "source" in result.node.metadata


@pytest.mark.unit
class TestLlamaIndexRetrieverEdgeCases:
    """Edge case tests for LlamaIndexRetriever."""

    def test_single_document(self, mock_embedding_model):
        """Test retrieval with a single document."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        chunks = [Document(text="single document about fasting", metadata={})]
        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=chunks)

        assert len(results) == 1
        assert "fasting" in results[0].node.get_content()

    def test_duplicate_documents(self, mock_embedding_model):
        """Test retrieval handles duplicate documents."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        chunks = [
            Document(text="fasting benefits", metadata={"id": "1"}),
            Document(text="fasting benefits", metadata={"id": "2"}),
        ]
        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query="fasting", chunks=chunks)

        # Should handle duplicates gracefully
        assert len(results) >= 1

    def test_very_long_query(self, sample_documents, mock_embedding_model):
        """Test retrieval with a very long query."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        long_query = "fasting " * 100  # Very long query
        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model)
        results = retriever.retrieve(query=long_query, chunks=sample_documents)

        # Should still return results
        assert len(results) > 0


@pytest.mark.unit
class TestLlamaIndexRetrieverConfiguration:
    """Test configuration options for LlamaIndexRetriever."""

    def test_custom_top_k(self, sample_documents, mock_embedding_model):
        """Test custom top_k parameter."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model, top_k=3)
        results = retriever.retrieve(query="health", chunks=sample_documents)

        assert len(results) <= 3

    def test_custom_rrf_k(self, sample_documents, mock_embedding_model):
        """Test custom RRF k parameter."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        # Should not raise with custom rrf_k
        retriever = LlamaIndexRetriever(
            embedding_model=mock_embedding_model, top_k=5, rrf_k=60
        )
        results = retriever.retrieve(query="fasting", chunks=sample_documents)

        assert len(results) > 0

    def test_retrieve_with_precomputed_embeddings(
        self, sample_documents, mock_embedding_model
    ):
        """Test retrieval with pre-computed embeddings."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model, top_k=5)

        # Pre-compute embeddings
        texts = [doc.get_content() for doc in sample_documents]
        embeddings = mock_embedding_model.get_text_embedding_batch(texts)

        results = retriever.retrieve_with_embeddings(
            query="fasting",
            chunks=sample_documents,
            chunk_embeddings=embeddings,
        )

        assert len(results) > 0
        assert all(isinstance(r, NodeWithScore) for r in results)

    def test_embedding_mismatch_raises_error(
        self, sample_documents, mock_embedding_model
    ):
        """Test that mismatched embedding count raises error."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(embedding_model=mock_embedding_model, top_k=5)

        # Create wrong number of embeddings
        embeddings = [[0.0] * 384]  # Only 1 embedding for 5 documents

        with pytest.raises(ValueError, match="Mismatch"):
            retriever.retrieve_with_embeddings(
                query="fasting",
                chunks=sample_documents,
                chunk_embeddings=embeddings,
            )
