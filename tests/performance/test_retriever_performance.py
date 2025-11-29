"""Performance tests for LlamaIndexRetriever.

These tests measure indexing and retrieval performance.
Marked as 'performance' to be skipped by default.

Run with: pytest -m performance tests/performance/test_retriever_performance.py

Performance Requirements:
- Indexing 200 chunks (with pre-computed embeddings): <1 second
- Retrieval latency (with pre-computed embeddings): <1 second

Note: Embedding computation is a separate concern from retrieval.
The retriever supports both on-the-fly embedding and pre-computed embeddings.
For production RAG pipelines, embeddings should be pre-computed for best performance.
"""

import random
import string
import time

import numpy as np
import pytest
from loguru import logger

from llama_index.core.schema import Document


def _make_scientific_paragraph(min_len: int = 100, max_len: int = 500) -> str:
    """Generate a synthetic scientific-like paragraph."""
    length = random.randint(min_len, max_len)
    words = []
    while sum(len(w) + 1 for w in words) < length:
        choice = random.random()
        if choice < 0.02:
            words.append("(et al., 2020)")
        elif choice < 0.04:
            words.append("e.g.,")
        elif choice < 0.06:
            words.append("i.e.,")
        else:
            wlen = random.randint(3, 12)
            word = "".join(random.choices(string.ascii_lowercase, k=wlen))
            words.append(word)
    return " ".join(words)[:length]


def _generate_test_documents(n: int = 200) -> list[Document]:
    """Generate n test documents for performance testing."""
    documents = []
    for i in range(n):
        text = _make_scientific_paragraph(min_len=200, max_len=600)
        # Add some keyword variety
        if i % 10 == 0:
            text = "fasting metabolism " + text
        elif i % 10 == 1:
            text = "machine learning neural network " + text
        elif i % 10 == 2:
            text = "protein synthesis amino acids " + text
        documents.append(
            Document(
                text=text,
                metadata={"doc_id": f"doc_{i}", "section": f"section_{i % 5}"},
            )
        )
    return documents


def _generate_mock_embeddings(n: int, dim: int = 384) -> list[list[float]]:
    """Generate mock embeddings for testing."""
    embeddings = []
    for i in range(n):
        np.random.seed(i)
        emb = np.random.rand(dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb.tolist())
    return embeddings


@pytest.fixture
def onnx_embedding_model():
    """Load the real ONNX embedding model for performance testing."""
    from config.onnx_embedding import ONNXEmbeddingModel

    return ONNXEmbeddingModel()


@pytest.fixture
def mock_embedding_model():
    """Fast mock embedding model for testing retrieval performance."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.embed_dim = 384

    def fake_embed(texts):
        """Generate deterministic embeddings instantly."""
        embeddings = []
        for i, text in enumerate(texts):
            np.random.seed(hash(text) % 2**32)
            emb = np.random.rand(384).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
        return embeddings

    mock.get_text_embedding_batch = fake_embed
    return mock


@pytest.fixture
def test_documents_200():
    """Generate 200 test documents."""
    random.seed(42)  # Reproducible
    return _generate_test_documents(200)


@pytest.fixture
def test_documents_50():
    """Generate 50 test documents for faster tests."""
    random.seed(42)
    return _generate_test_documents(50)


@pytest.mark.performance
class TestRetrieverPerformance:
    """Performance benchmarks for LlamaIndexRetriever."""

    def test_indexing_200_chunks_with_precomputed_embeddings_under_1_second(
        self, mock_embedding_model, test_documents_200
    ):
        """Test that indexing + retrieval with pre-computed embeddings takes <1 second.

        This is the realistic production scenario where embeddings are computed
        once during document processing and reused for multiple queries.
        """
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(
            embedding_model=mock_embedding_model,
            top_k=10,
        )

        # Pre-compute embeddings (simulates production pipeline)
        embeddings = _generate_mock_embeddings(len(test_documents_200))

        # Time only the retrieval (not embedding computation)
        start = time.perf_counter()
        results = retriever.retrieve_with_embeddings(
            query="fasting metabolism",
            chunks=test_documents_200,
            chunk_embeddings=embeddings,
        )
        elapsed = time.perf_counter() - start

        logger.info(
            f"Retrieval of 200 chunks (pre-computed embeddings) took {elapsed:.3f}s"
        )

        assert len(results) > 0
        assert elapsed < 1.0, f"Retrieval took {elapsed:.3f}s, expected <1s"

    def test_retrieval_latency_under_1_second(
        self, mock_embedding_model, test_documents_50
    ):
        """Test that retrieval latency is under 1 second."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(
            embedding_model=mock_embedding_model,
            top_k=5,
        )

        # Pre-compute embeddings
        embeddings = _generate_mock_embeddings(len(test_documents_50))

        start = time.perf_counter()
        results = retriever.retrieve_with_embeddings(
            query="machine learning",
            chunks=test_documents_50,
            chunk_embeddings=embeddings,
        )
        elapsed = time.perf_counter() - start

        logger.info(f"Retrieval of 50 chunks took {elapsed:.3f}s")

        assert len(results) > 0
        assert elapsed < 1.0, f"Retrieval took {elapsed:.3f}s, expected <1s"

    def test_multiple_queries_consistent_performance(
        self, mock_embedding_model, test_documents_50
    ):
        """Test that multiple queries have consistent performance."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(
            embedding_model=mock_embedding_model,
            top_k=5,
        )

        # Pre-compute embeddings once
        embeddings = _generate_mock_embeddings(len(test_documents_50))

        queries = [
            "fasting benefits",
            "machine learning",
            "protein synthesis",
            "neural network architecture",
            "metabolic pathway",
        ]

        times = []
        for query in queries:
            start = time.perf_counter()
            results = retriever.retrieve_with_embeddings(
                query=query,
                chunks=test_documents_50,
                chunk_embeddings=embeddings,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert len(results) > 0

        avg_time = sum(times) / len(times)
        max_time = max(times)

        logger.info(f"Multiple queries: avg={avg_time:.3f}s, max={max_time:.3f}s")

        # All queries should complete in reasonable time
        assert max_time < 1.0, f"Slowest query took {max_time:.3f}s"

    def test_full_pipeline_with_real_embeddings(
        self, onnx_embedding_model, test_documents_50
    ):
        """Test full pipeline including real embedding computation.

        This tests the complete flow with real ONNX embeddings.
        Expected to be slower due to embedding computation.
        """
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(
            embedding_model=onnx_embedding_model,
            top_k=5,
        )

        start = time.perf_counter()
        results = retriever.retrieve(query="fasting", chunks=test_documents_50)
        elapsed = time.perf_counter() - start

        logger.info(f"Full pipeline (50 chunks, real embeddings) took {elapsed:.3f}s")

        assert len(results) > 0
        # More relaxed timing for full pipeline with real embeddings
        assert elapsed < 5.0, f"Full pipeline took {elapsed:.3f}s, expected <5s"


@pytest.mark.performance
class TestHybridSearchQuality:
    """Test the quality of hybrid search results."""

    def test_keyword_match_ranks_higher(self, onnx_embedding_model):
        """Test that exact keyword matches rank higher with hybrid search."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        # Create documents with clear keyword distinctions
        documents = [
            Document(
                text="Intermittent fasting improves metabolic health", metadata={}
            ),
            Document(text="Exercise and cardiovascular fitness", metadata={}),
            Document(text="The benefits of fasting for longevity", metadata={}),
            Document(text="Protein intake and muscle synthesis", metadata={}),
            Document(text="Fasting protocols and autophagy activation", metadata={}),
        ]

        retriever = LlamaIndexRetriever(
            embedding_model=onnx_embedding_model,
            top_k=5,
        )

        results = retriever.retrieve(query="fasting", chunks=documents)

        # Get texts of top 3 results
        top_texts = [r.node.get_content().lower() for r in results[:3]]

        # At least 2 of top 3 should contain "fasting"
        fasting_count = sum(1 for t in top_texts if "fasting" in t)
        assert (
            fasting_count >= 2
        ), f"Expected at least 2 fasting docs in top 3, got {fasting_count}"

    def test_semantic_similarity_contributes(self, onnx_embedding_model):
        """Test that semantic similarity contributes to ranking."""
        from retrieval.llamaindex_retriever import LlamaIndexRetriever

        # Create documents where semantic meaning matters
        documents = [
            Document(
                text="Caloric restriction extends lifespan", metadata={}
            ),  # Semantically similar to fasting
            Document(text="Random unrelated topic about weather", metadata={}),
            Document(
                text="Intermittent fasting protocol", metadata={}
            ),  # Direct keyword match
            Document(
                text="Food abstinence improves health", metadata={}
            ),  # Semantically similar
        ]

        retriever = LlamaIndexRetriever(
            embedding_model=onnx_embedding_model,
            top_k=4,
        )

        results = retriever.retrieve(query="fasting benefits", chunks=documents)

        # Weather doc should not be in top 2
        top_2_texts = [r.node.get_content().lower() for r in results[:2]]
        assert not any("weather" in t for t in top_2_texts)
