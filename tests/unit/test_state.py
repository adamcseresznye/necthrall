import re
from datetime import datetime

import pytest
from pydantic import ValidationError

from models.state import State


@pytest.mark.unit
def test_state_minimal_init():
    s = State(query="test query")
    assert s.query == "test query"
    # request_id present and looks like a uuid string
    assert isinstance(s.request_id, str)
    assert re.match(r"[0-9a-fA-F\-]{36}", s.request_id)
    assert isinstance(s.timestamp, datetime)
    # Optional fields default to None or empty list for errors
    assert s.optimized_query is None
    assert s.papers == []
    assert s.errors == []


@pytest.mark.unit
def test_progressive_updates():
    s = State(query="q")
    s.update_fields(optimized_query="q optimized")
    assert s.optimized_query == "q optimized"

    # simulate retrieval
    papers = [{"paperId": "p1", "title": "Paper 1"}]
    s.update_fields(papers=papers)
    assert len(s.papers) == 1
    # pydantic converts dicts to `Paper` models; verify fields preserved
    assert s.papers[0].paperId == "p1"
    assert s.papers[0].title == "Paper 1"

    # ranking
    ranked = [{"paperId": "p1", "title": "Paper 1", "score": 0.9}]
    s.update_fields(ranked_papers=ranked)
    assert len(s.ranked_papers) == 1
    rp = s.ranked_papers[0].model_dump()
    assert rp.get("paperId") == "p1"
    assert rp.get("score") == 0.9


@pytest.mark.unit
def test_invalid_types_raise():
    with pytest.raises(ValidationError):
        # query is required and must be a str; passing an int should fail
        State(query=123)  # type: ignore


@pytest.mark.unit
def test_error_tracking_and_defaults():
    s = State(query="err")
    s.append_error("first error")
    assert s.errors == ["first error"]

    # defaults regenerated per instance
    s2 = State(query="err2")
    assert s2.request_id != s.request_id


@pytest.mark.unit
def test_synthesis_fields_defaults():
    """Test that answer and citations fields default correctly."""
    s = State(query="synthesis test")
    # answer defaults to None
    assert s.answer is None
    # citations defaults to empty list
    assert s.citations == []
    assert isinstance(s.citations, list)


@pytest.mark.unit
def test_synthesis_fields_assignment():
    """Test that answer and citations can be assigned and updated."""
    s = State(query="synthesis test")

    # Set answer
    s.update_fields(answer="The evidence suggests...")
    assert s.answer == "The evidence suggests..."

    # Set citations as list of integers (paper indices)
    s.update_fields(citations=[0, 2, 5])
    assert s.citations == [0, 2, 5]

    # Verify other existing fields remain intact
    assert s.finalists == []
    assert s.passages == []
