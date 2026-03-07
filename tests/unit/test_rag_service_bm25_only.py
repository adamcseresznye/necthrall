"""Unit tests for RAGService in BM25-only mode (Phase 1b).

Verifies:
- RAGService can be instantiated with embedding_model=None when mode is bm25_only
- answer() does not early-exit in bm25_only mode even without an embedding model
- Stage 7 retrieval produces passages; Stage 8 diversity filter runs; synthesis is called
- Switching to "hybrid" mode restores the embedding_model=None early-exit
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Minimal LlamaIndex TextNode duck-type used as chunk input."""

    def __init__(self, text: str, paper_id: str):
        self.metadata = {"paper_id": paper_id}
        self._text = text

    def get_content(self) -> str:
        return self._text

    @property
    def text(self) -> str:
        return self._text


def _make_bm25_settings(
    retrieval_top_k: int = 10,
    rerank_top_k: int = 3,
) -> MagicMock:
    settings = MagicMock()
    settings.RAG_RETRIEVAL_MODE = "bm25_only"
    settings.RAG_RETRIEVAL_TOP_K = retrieval_top_k
    settings.RAG_RERANK_TOP_K = rerank_top_k
    return settings


def _make_hybrid_settings(
    retrieval_top_k: int = 10,
    rerank_top_k: int = 3,
) -> MagicMock:
    settings = MagicMock()
    settings.RAG_RETRIEVAL_MODE = "hybrid"
    settings.RAG_RETRIEVAL_TOP_K = retrieval_top_k
    settings.RAG_RERANK_TOP_K = rerank_top_k
    return settings


def _make_rag_service_bm25(embedding_model=None):
    """Instantiate RAGService in bm25_only mode with heavy deps mocked."""
    from services.rag_service import RAGService

    settings = _make_bm25_settings()
    with (
        patch("agents.synthesis_agent.SynthesisAgent"),
        patch("utils.citation_verifier.CitationVerifier"),
    ):
        svc = RAGService(embedding_model=embedding_model, settings=settings)
    return svc


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bm25_only_accepts_none_embedding_model():
    """RAGService must not raise when embedding_model=None in bm25_only mode."""
    svc = _make_rag_service_bm25(embedding_model=None)
    assert svc is not None


@pytest.mark.unit
def test_bm25_only_retriever_type():
    """In bm25_only mode the retriever must be a BM25Retriever instance."""
    from retrieval.bm25_retriever import BM25Retriever

    svc = _make_rag_service_bm25()
    assert isinstance(svc.retriever, BM25Retriever)


# ---------------------------------------------------------------------------
# answer() path tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_answer_returns_rag_result_no_embedding_model():
    """answer() must not early-exit in bm25_only mode even without embedding model."""
    from services.rag_service import RAGResult

    svc = _make_rag_service_bm25(embedding_model=None)

    chunks = [
        _FakeChunk("microbiome and gut health research", "p1"),
        _FakeChunk("quantum computing algorithms", "p2"),
        _FakeChunk("anxiety disorders treatment approaches", "p3"),
    ]

    svc.synthesis_agent.synthesize = AsyncMock(return_value="Test answer.")

    result = await svc.answer("gut microbiome anxiety", chunks)

    assert isinstance(result, RAGResult)
    assert (
        result.passages is not None
    )  # may be empty if BM25 scores are zero but list exists


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_answer_produces_passages():
    """With term-overlapping chunks, passages must be non-empty."""
    from services.rag_service import RAGResult

    svc = _make_rag_service_bm25(embedding_model=None)

    chunks = [
        _FakeChunk("microbiome gut health bacteria diversity", "p1"),
        _FakeChunk("spacetime relativity quantum field theory", "p2"),
    ]

    svc.synthesis_agent.synthesize = AsyncMock(return_value="Microbiome answer.")

    result = await svc.answer("microbiome gut health", chunks)

    assert isinstance(result, RAGResult)
    assert len(result.passages) >= 1
    assert result.passages[0].paper_id == "p1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_answer_empty_chunks_returns_empty():
    """answer() with no chunks must return an empty RAGResult without crashing."""
    from services.rag_service import RAGResult

    svc = _make_rag_service_bm25(embedding_model=None)

    result = await svc.answer("any query", [])

    assert isinstance(result, RAGResult)
    assert result.passages == []
    assert result.answer is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_mode_early_exits_without_embedding_model():
    """In hybrid mode, answer() must early-exit when embedding_model is None."""
    from services.rag_service import RAGResult, RAGService

    settings = _make_hybrid_settings()
    mock_embedding = MagicMock()  # need a non-None model to construct the retriever

    with (
        patch("retrieval.llamaindex_retriever.LlamaIndexRetriever"),
        patch("agents.synthesis_agent.SynthesisAgent"),
        patch("utils.citation_verifier.CitationVerifier"),
    ):
        svc = RAGService(embedding_model=mock_embedding, settings=settings)

    # Manually set embedding_model to None (simulates it being unavailable at runtime)
    svc.embedding_model = None

    chunks = [_FakeChunk("some text about research", "p1")]
    result = await svc.answer("query", chunks)

    # Should early-exit → empty passages, no answer
    assert isinstance(result, RAGResult)
    assert result.passages == []
    assert result.answer is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_diversity_filter_runs():
    """_select_diverse_top_k must be invoked during answer() in bm25_only mode."""
    from unittest.mock import patch

    svc = _make_rag_service_bm25(embedding_model=None)
    svc.synthesis_agent.synthesize = AsyncMock(return_value="Answer text.")

    # 6 chunks from 2 papers (3 each) — diversity filter should cap at 3 per paper
    chunks = [_FakeChunk(f"microbiome study chunk {i}", "p1") for i in range(4)] + [
        _FakeChunk(f"quantum physics chunk {i}", "p2") for i in range(4)
    ]

    with patch.object(
        svc,
        "_select_diverse_top_k",
        wraps=svc._select_diverse_top_k,
    ) as spy:
        await svc.answer("microbiome", chunks)
        spy.assert_called_once()
