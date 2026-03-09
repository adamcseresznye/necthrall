"""Unit tests for RAGService after Phase 1 (reranker removed).

Verifies:
- No reranker attribute on RAGService
- _select_diverse_top_k() is called directly after retrieval
- AcquisitionAgent TARGET_PDF_COUNT == 3 and candidate pool sliced to 9
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agents.acquisition_agent as acquisition_mod
from services.rag_service import RAGService

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_node(paper_id: str = "paper_1", score: float = 0.9) -> MagicMock:
    """Return a mock NodeWithScore-like object with the metadata shape used by RAGService."""
    node = MagicMock()
    node.metadata = {"paper_id": paper_id, "title": "Test Paper"}
    node.get_content.return_value = "Some passage text."

    passage = MagicMock()
    passage.node = node
    passage.score = score
    return passage


def _make_settings(retrieval_top_k: int = 50, passages_top_k: int = 12) -> MagicMock:
    settings = MagicMock()
    settings.RAG_RETRIEVAL_TOP_K = retrieval_top_k
    settings.RAG_PASSAGES_TOP_K = passages_top_k
    return settings


def _make_rag_service() -> RAGService:
    """Instantiate RAGService with all heavy dependencies mocked out."""
    mock_embedding = MagicMock()
    mock_settings = _make_settings()

    with (
        patch("retrieval.llamaindex_retriever.LlamaIndexRetriever"),
        patch("agents.synthesis_agent.SynthesisAgent"),
        patch("utils.citation_verifier.CitationVerifier"),
    ):
        service = RAGService(embedding_model=mock_embedding, settings=mock_settings)

    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_service_has_no_reranker():
    """RAGService should not instantiate a reranker after Phase 1."""
    service = _make_rag_service()
    assert not hasattr(
        service, "reranker"
    ), "RAGService must not have a 'reranker' attribute after Phase 1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_answer_calls_diverse_top_k_directly():
    """After retrieval _select_diverse_top_k should be called exactly once (no reranking step)."""
    service = _make_rag_service()

    # Build 3 fake passages from different papers
    sample_passages = [
        _make_mock_node(paper_id=f"paper_{i}", score=1.0 - i * 0.1) for i in range(3)
    ]

    # Mock the retriever to synchronously return sample_passages (asyncio.to_thread wraps it)
    service.retriever.retrieve = MagicMock(return_value=sample_passages)

    # Mock synthesis and verification
    service.synthesis_agent.synthesize = AsyncMock(return_value="Mock answer.")
    service.verifier.verify_citations = AsyncMock(return_value={"verified": True})

    sample_chunks = [MagicMock()]  # non-empty so retrieval stage is entered

    with patch.object(
        service,
        "_select_diverse_top_k",
        wraps=service._select_diverse_top_k,
    ) as spy:
        await service.answer("test query", sample_chunks)
        spy.assert_called_once()


@pytest.mark.unit
def test_acquisition_agent_pdf_target_is_3():
    """TARGET_PDF_COUNT must equal 3 after Phase 1."""
    src = inspect.getsource(acquisition_mod.AcquisitionAgent.process)
    assert (
        "TARGET_PDF_COUNT = 3" in src
    ), "Expected 'TARGET_PDF_COUNT = 3' in AcquisitionAgent.process source"


@pytest.mark.unit
def test_acquisition_agent_candidate_slice_is_9():
    """PDF candidate pool must be sliced to [:9] after Phase 1."""
    src = inspect.getsource(acquisition_mod.AcquisitionAgent.process)
    assert "][:9]" in src, "Expected '][:9]' in AcquisitionAgent.process source"
