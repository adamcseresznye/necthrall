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
    assert s.papers is None
    assert s.errors == []


@pytest.mark.unit
def test_progressive_updates():
    s = State(query="q")
    s.update_fields(optimized_query="q optimized")
    assert s.optimized_query == "q optimized"

    # simulate retrieval
    papers = [{"id": "p1", "title": "Paper 1"}]
    s.update_fields(papers=papers)
    assert s.papers == papers

    # ranking
    ranked = [{"id": "p1", "score": 0.9}]
    s.update_fields(ranked_papers=ranked)
    assert s.ranked_papers == ranked


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
