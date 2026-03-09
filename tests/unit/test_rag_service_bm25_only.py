"""Unit tests for RAGService in BM25-only mode.

Verifies:
- RAGService can be instantiated with settings only
- answer() produces passages; diversity filter runs; synthesis is called
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
    passages_top_k: int = 3,
) -> MagicMock:
    settings = MagicMock()
    settings.RAG_RETRIEVAL_TOP_K = retrieval_top_k
    settings.RAG_PASSAGES_TOP_K = passages_top_k
    return settings


def _make_rag_service_bm25():
    """Instantiate RAGService with heavy deps mocked."""
    from services.rag_service import RAGService

    settings = _make_bm25_settings()
    with (
        patch("agents.synthesis_agent.SynthesisAgent"),
        patch("utils.citation_verifier.CitationVerifier"),
    ):
        svc = RAGService(settings=settings)
    return svc


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bm25_only_retriever_type():
    """RAGService must use a BM25Retriever instance."""
    from retrieval.bm25_retriever import BM25Retriever

    svc = _make_rag_service_bm25()
    assert isinstance(svc.retriever, BM25Retriever)


# ---------------------------------------------------------------------------
# answer() path tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_answer_returns_rag_result():
    """answer() returns a RAGResult."""
    from services.rag_service import RAGResult

    svc = _make_rag_service_bm25()

    chunks = [
        _FakeChunk("microbiome and gut health research", "p1"),
        _FakeChunk("quantum computing algorithms", "p2"),
        _FakeChunk("anxiety disorders treatment approaches", "p3"),
    ]

    svc.synthesis_agent.synthesize = AsyncMock(return_value="Test answer.")

    result = await svc.answer("gut microbiome anxiety", chunks)

    assert isinstance(result, RAGResult)
    assert result.passages is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_answer_produces_passages():
    """With term-overlapping chunks, passages must be non-empty."""
    from services.rag_service import RAGResult

    svc = _make_rag_service_bm25()

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

    svc = _make_rag_service_bm25()

    result = await svc.answer("any query", [])

    assert isinstance(result, RAGResult)
    assert result.passages == []
    assert result.answer is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_only_diversity_filter_runs():
    """_select_diverse_top_k must be invoked during answer()."""
    from unittest.mock import patch

    svc = _make_rag_service_bm25()
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
