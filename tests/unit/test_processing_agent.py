import pytest

from agents.processing_agent import ProcessingAgent
from models.state import State


@pytest.fixture
def make_state():
    def _make(query: str, passages):
        return State(query=query, passages=passages)

    return _make


@pytest.mark.unit
def test_processing_enriches_state_single_passage(make_state):
    state = make_state(
        "test",
        [
            {
                "paperId": "abc123",
                "title": "Test Paper",
                "text": "# Introduction\n"
                + "A " * 5000
                + "\n# Methods\n"
                + "B " * 5000,
                "text_source": "pdf",
                "year": 2020,
                "venue": "Journal of Testing",
                "influentialCitationCount": 2,
            }
        ],
    )

    agent = ProcessingAgent(chunk_size=500, chunk_overlap=50)
    updated_state = agent.process(state)

    assert hasattr(updated_state, "chunks")
    assert updated_state.chunks is not None
    assert len(updated_state.chunks) >= 10
    first_meta = updated_state.chunks[0].metadata
    assert first_meta["paper_id"] == "abc123"
    assert first_meta["paper_title"] == "Test Paper"
    assert first_meta["year"] == 2020
    assert first_meta["venue"] == "Journal of Testing"
    assert first_meta["influential_citation_count"] == 2


@pytest.mark.unit
def test_processing_multiple_passages_generates_chunks(make_state):
    passages = []
    for i in range(3):
        passages.append(
            {
                "paperId": f"p{i}",
                "title": f"Paper {i}",
                "text": "# Introduction\n"
                + ("X " * 4000)
                + "\n# Results\n"
                + ("Y " * 4000),
                "text_source": "pdf",
                "year": 2019 + i,
                "venue": "Conf",
                "influentialCitationCount": i,
            }
        )

    state = make_state("multi", passages)
    agent = ProcessingAgent(chunk_size=500, chunk_overlap=50)
    updated = agent.process(state)

    assert hasattr(updated, "chunks")
    assert len(updated.chunks) >= 30


@pytest.mark.unit
def test_empty_passage_is_skipped(make_state):
    state = make_state("empty", [{"paperId": "e1", "text": "", "text_source": "pdf"}])
    agent = ProcessingAgent(chunk_size=200, chunk_overlap=20)
    updated = agent.process(state)

    assert hasattr(updated, "chunks")
    assert isinstance(updated.chunks, list)
    assert len(updated.chunks) == 0
    # No critical error appended (empty passage skipped)
    assert all("Zero chunks" not in e for e in updated.errors)


@pytest.mark.unit
def test_markdown_parser_handles_plain_text(make_state):
    # No explicit Markdown headers -> MarkdownNodeParser chunks by size
    long_text = "This is a paper without headers. " + ("word " * 3000)
    state = make_state(
        "plaintext", [{"paperId": "f1", "text": long_text, "text_source": "pdf"}]
    )
    agent = ProcessingAgent(chunk_size=400, chunk_overlap=50)
    updated = agent.process(state)

    assert hasattr(updated, "chunks")
    assert len(updated.chunks) > 0


@pytest.mark.unit
def test_no_passages_triggers_error():
    state = State(query="none", passages=None)
    agent = ProcessingAgent()
    res = agent.process(state)
    assert len(res.errors) > 0
    assert any(
        "No passages available" in e or "Zero chunks generated" in e for e in res.errors
    )
