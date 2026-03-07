"""Unit tests for BM25Retriever.

These tests are self-contained and require no external services or models.
"""

import pytest

from retrieval.bm25_retriever import BM25Retriever, SimpleNode, SimpleNodeWithScore

# ---------------------------------------------------------------------------
# Minimal chunk fakes that mimic the LlamaIndex TextNode interface
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Duck-typed stand-in for a LlamaIndex TextNode."""

    def __init__(self, text: str, paper_id: str):
        self.metadata = {"paper_id": paper_id}
        self._text = text

    def get_content(self) -> str:
        return self._text

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bm25_retriever_prefers_exact_terms():
    """The chunk whose text overlaps most with the query should rank first."""
    chunks = [
        _FakeChunk("microbiome affects anxiety via vagus nerve", "p1"),
        _FakeChunk("quantum entanglement and spacetime curvature", "p2"),
    ]
    r = BM25Retriever(top_k=2)
    results = r.retrieve("gut microbiome anxiety", chunks)

    assert len(results) == 2
    assert (
        results[0].node.metadata["paper_id"] == "p1"
    ), "Expected the microbiome chunk to rank first"


@pytest.mark.unit
def test_bm25_retriever_top_k_respected():
    """BM25Retriever must return at most top_k results."""
    chunks = [
        _FakeChunk(f"document about topic number {i}", f"p{i}") for i in range(10)
    ]
    r = BM25Retriever(top_k=3)
    results = r.retrieve("topic document", chunks)

    assert len(results) == 3


@pytest.mark.unit
def test_bm25_retriever_top_k_larger_than_corpus():
    """When top_k exceeds the corpus size, all chunks are returned."""
    chunks = [
        _FakeChunk("neural networks deep learning", "p1"),
        _FakeChunk("protein folding structure prediction", "p2"),
    ]
    r = BM25Retriever(top_k=10)
    results = r.retrieve("neural networks", chunks)

    assert len(results) == 2


@pytest.mark.unit
def test_bm25_retriever_all_zero_scores_returns_results():
    """When there is no term overlap the retriever must still return results."""
    chunks = [
        _FakeChunk("zyxwVUTS irrelevant words here", "p1"),
        _FakeChunk("abcdefgh completely unrelated text", "p2"),
    ]
    r = BM25Retriever(top_k=2)
    results = r.retrieve("completelydifferentquery", chunks)

    # Must not return empty list even with zero BM25 scores
    assert len(results) > 0


@pytest.mark.unit
def test_bm25_retriever_empty_chunks_returns_empty():
    """An empty chunk list should produce an empty result without error."""
    r = BM25Retriever(top_k=5)
    results = r.retrieve("any query", [])

    assert results == []


@pytest.mark.unit
def test_bm25_retriever_result_types():
    """Returned objects must satisfy the NodeWithScore-compatible interface."""
    chunks = [_FakeChunk("sample text about biology", "bio_paper")]
    r = BM25Retriever(top_k=1)
    results = r.retrieve("biology", chunks)

    assert len(results) == 1
    item = results[0]

    # SimpleNodeWithScore interface
    assert isinstance(item, SimpleNodeWithScore)
    assert isinstance(item.node, SimpleNode)
    assert isinstance(item.score, float)

    # SimpleNode interface
    assert item.node.get_content() == "sample text about biology"
    assert item.node.text == "sample text about biology"
    assert item.node.metadata == {"paper_id": "bio_paper"}


@pytest.mark.unit
def test_bm25_retriever_metadata_preserved():
    """All metadata fields of the original chunk must be preserved in results."""

    class _RichChunk:
        def __init__(self):
            self.metadata = {
                "paper_id": "rich_1",
                "paper_title": "Some Title",
                "citation_count": 42,
                "year": 2024,
            }

        def get_content(self) -> str:
            return "rich content about neuroscience"

    r = BM25Retriever(top_k=1)
    results = r.retrieve("neuroscience", [_RichChunk()])

    assert results[0].node.metadata["paper_title"] == "Some Title"
    assert results[0].node.metadata["citation_count"] == 42
    assert results[0].node.metadata["year"] == 2024
