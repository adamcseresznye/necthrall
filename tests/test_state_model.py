import pytest
from pydantic import ValidationError
from models.state import State, Paper, Passage, Score


def test_state_creation():
    """Tests basic State object creation with new optimized fields."""
    state = State(query="test query")
    assert state.original_query == "test query"
    assert state.query == "test query"  # Test alias backward compatibility
    assert isinstance(state.request_id, str)
    assert state.papers_metadata == []
    assert state.filtered_papers == []
    assert state.optimized_query is None
    assert state.filtering_scores is None
    assert state.dedup_stats is None
    assert state.search_quality is None
    assert state.refinement_count == 0


def test_state_creation_with_original_query():
    """Tests State creation using original_query field directly."""
    state = State(original_query="direct query test")
    assert state.original_query == "direct query test"
    assert state.query == "direct query test"  # Alias should work both ways


def test_state_with_papers():
    """Tests State object creation with paper metadata."""
    papers = [
        Paper(
            paper_id="1",
            title="Paper 1",
            authors=["Author 1"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url="http://example.com/1.pdf",
            type="article",
        ),
        Paper(
            paper_id="2",
            title="Paper 2",
            authors=["Author 2"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url="http://example.com/2.pdf",
            type="review",
        ),
    ]
    state = State(query="test", papers_metadata=papers)
    assert len(state.papers_metadata) == 2
    assert state.papers_metadata[0].pdf_url is not None


def test_state_validator_no_pdf_urls():
    """Tests the validator when no papers have PDF URLs."""
    papers = [
        Paper(
            paper_id="1",
            title="Paper 1",
            authors=["Author 1"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url=None,
            type="article",
        )
    ]
    state = State(query="test", papers_metadata=papers)
    assert "No papers with PDF URLs found." in state.validation_errors


def test_state_validator_empty_papers_list():
    """Tests the validator with an empty list of papers."""
    state = State(query="test", papers_metadata=[])
    assert state.papers_metadata == []


def test_state_serialization():
    """Tests that the State object can be serialized to a dict."""
    papers = [
        Paper(
            paper_id="1",
            title="Paper 1",
            authors=["Author 1"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url="http://example.com/1.pdf",
            type="article",
        )
    ]
    state = State(query="test", papers_metadata=papers)
    state_dict = state.model_dump()
    # Default serialization uses field names, not aliases
    assert state_dict["original_query"] == "test"
    assert "query" not in state_dict  # Alias not used by default
    assert len(state_dict["papers_metadata"]) == 1
    assert state_dict["papers_metadata"][0]["pdf_url"] == "http://example.com/1.pdf"


def test_state_serialization_by_alias():
    """Tests serialization with by_alias parameter."""
    state = State(query="test")
    # Test serialization with alias (legacy format)
    state_dict = state.model_dump(by_alias=True)
    assert "query" in state_dict
    assert "original_query" not in state_dict

    # Test serialization without alias (new format)
    state_dict = state.model_dump(by_alias=False)
    assert "original_query" in state_dict
    assert "query" not in state_dict


def test_optimized_query_validation():
    """Test optimized_query field validation enforces length constraints."""
    state = State(query="test")

    # Valid optimized query
    state.optimized_query = "CRISPR-Cas9 off-target effects in mammalian cells"
    assert state.optimized_query is not None

    # Invalid: too short
    with pytest.raises(ValueError, match="20-200 characters"):
        state.optimized_query = "short"

    # Invalid: too long
    with pytest.raises(ValueError, match="20-200 characters"):
        state.optimized_query = "x" * 201


def test_query_length_validation():
    """Test query length validation for both original and optimized queries."""
    # Valid query
    state = State(query="This is a valid query with sufficient length")
    assert state.original_query is not None

    # Invalid: too short
    with pytest.raises(ValueError, match="at least 3 characters"):
        State(query="hi")

    # Invalid: too long
    with pytest.raises(ValueError, match="less than 256 characters"):
        State(query="x" * 257)

    # Invalid: just whitespace - this will fail length validation first (3 chars required)
    with pytest.raises(ValueError, match="at least 3 characters"):
        State(query="   ")


def test_filtered_papers_count_validation():
    """Test filtered_papers cannot exceed 25 papers."""
    state = State(query="test")

    # Valid: exactly 25 papers
    state.filtered_papers = [
        Paper(
            paper_id=str(i),
            title=f"Paper {i}",
            authors=[f"Author {i}"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
        )
        for i in range(25)
    ]
    assert len(state.filtered_papers) == 25

    # Invalid: more than 25
    with pytest.raises(ValueError, match="≤25"):
        state.filtered_papers = [
            Paper(
                paper_id=str(i),
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                year=2023,
                journal="N/A",
                citation_count=0,
                pdf_url=f"https://example.com/{i}.pdf",
                type="article",
            )
            for i in range(26)
        ]


def test_search_quality_structure():
    """Test search_quality dict has correct structure."""
    state = State(
        query="test",
        search_quality={
            "passed": True,
            "reason": "Found 78 papers with avg_relevance 0.65",
            "paper_count": 78,
            "avg_relevance": 0.65,
        },
    )

    assert state.search_quality["passed"] is True
    assert state.search_quality["paper_count"] == 78
    assert 0 <= state.search_quality["avg_relevance"] <= 1

    # Test validation failures
    with pytest.raises(
        ValueError, match="must contain: passed, reason, paper_count, avg_relevance"
    ):
        State(query="test", search_quality={"incomplete": "dict"})

    with pytest.raises(ValueError, match="must be boolean"):
        State(
            query="test",
            search_quality={
                "passed": "not_boolean",
                "reason": "test",
                "paper_count": 1,
                "avg_relevance": 0.5,
            },
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        State(
            query="test",
            search_quality={
                "passed": True,
                "reason": "test",
                "paper_count": 1,
                "avg_relevance": 1.5,
            },
        )


def test_dedup_stats_structure():
    """Test dedup_stats tracks deduplication correctly."""
    state = State(
        query="test",
        dedup_stats={"raw_count": 200, "unique_count": 145, "duplicates_removed": 55},
    )

    assert state.dedup_stats["raw_count"] == 200
    assert state.dedup_stats["unique_count"] == 145
    assert state.dedup_stats["duplicates_removed"] == 55

    # Test validation failures - missing required keys
    with pytest.raises(
        ValueError, match="must contain: raw_count, unique_count, duplicates_removed"
    ):
        State(
            query="test", dedup_stats={"raw_count": 100}
        )  # Missing other required keys

    with pytest.raises(ValueError, match="cannot be greater than raw_count"):
        State(
            query="test",
            dedup_stats={
                "raw_count": 100,
                "unique_count": 150,
                "duplicates_removed": 0,
            },
        )


def test_filtering_scores_structure():
    """Test filtering_scores has expected structure."""
    state = State(
        query="test",
        filtering_scores={
            "bm25_top_50": ["paper1", "paper2"],
            "semantic_top_25": ["paper1", "paper3"],
            "avg_bm25_score": 8.2,
            "avg_semantic_score": 0.78,
        },
    )

    assert "bm25_top_50" in state.filtering_scores
    assert "avg_bm25_score" in state.filtering_scores

    # Test validation - should pass with at least one expected key
    state2 = State(query="test", filtering_scores={"avg_bm25_score": 7.5})
    assert state2.filtering_scores is not None

    # Test validation failure
    with pytest.raises(ValueError, match="must contain at least one of"):
        State(query="test", filtering_scores={"unknown_key": "value"})


def test_passage_and_score_models():
    """Test the new Passage and Score models work correctly."""
    passage = Passage(
        passage_id="ps1",
        paper_id="p1",
        text="This is a test passage.",
        page_number=1,
        char_start=0,
        char_end=25,
    )

    assert passage.passage_id == "ps1"
    assert passage.paper_id == "p1"
    assert passage.text == "This is a test passage."

    score = Score(
        score_type="relevance",
        value=0.95,
        justification="Highly relevant to query",
        confidence=0.9,
    )

    assert score.score_type == "relevance"
    assert score.value == 0.95
    assert score.justification == "Highly relevant to query"


def test_backward_compatibility():
    """Test old State usage patterns still work."""
    # Simulate old code that uses 'query' field
    state = State(query="What is intermittent fasting?")

    # Both alias and field should work
    assert state.query == state.original_query
    assert state.original_query == "What is intermittent fasting?"

    # Test that old serialization still works
    state_dict = state.model_dump(by_alias=True)
    assert state_dict["query"] == "What is intermittent fasting?"
    assert "original_query" not in state_dict


def test_state_flows_through_optimized_pipeline():
    """Test State correctly tracks data through new optimized flow."""
    state = State(query="fasting benefits")

    # After QueryOptimizationAgent
    state.optimized_query = "metabolic effects of intermittent fasting protocols"
    assert state.optimized_query is not None

    # After SearchAgent (100 papers)
    state.papers_metadata = [
        Paper(
            paper_id=str(i),
            title=f"Paper {i}",
            authors=[f"Author {i}"],
            year=2023,
            journal="N/A",
            citation_count=0,
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
        )
        for i in range(100)
    ]
    state.search_quality = {
        "passed": True,
        "reason": "Found 100 papers",
        "paper_count": 100,
        "avg_relevance": 0.72,
    }

    # After DeduplicationAgent
    state.dedup_stats = {"raw_count": 100, "unique_count": 95, "duplicates_removed": 5}

    # After FilteringAgent
    state.filtered_papers = state.papers_metadata[:25]
    state.filtering_scores = {"avg_bm25_score": 8.2, "avg_semantic_score": 0.78}

    assert len(state.filtered_papers) == 25
    assert state.dedup_stats["duplicates_removed"] == 5
    assert state.search_quality["paper_count"] == 100
