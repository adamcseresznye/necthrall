"""Unit tests for quality_gate.validate_quality function."""

import pytest

from agents.quality_gate import validate_quality


@pytest.fixture
def mock_paper_with_all_fields():
    """A mock paper with all required fields."""
    return {
        "paperId": "test_paper_1",
        "abstract": "This is a test abstract.",
    }


@pytest.fixture
def mock_paper_missing_abstract():
    """A mock paper missing abstract."""
    return {
        "paperId": "test_paper_2",
    }


@pytest.mark.unit
def test_valid_paper_set_passes(mock_paper_with_all_fields, mock_paper_missing_abstract):
    """Test case 1: Valid paper set with all criteria met passes validation."""
    papers = [mock_paper_with_all_fields] * 25  # 25 papers with all fields
    papers.extend([mock_paper_missing_abstract] * 5)  # 5 more, total 30

    result = validate_quality(papers)

    assert result["passed"] is True
    assert result["metrics"]["paper_count"] == 30
    assert result["metrics"]["abstract_coverage"] == 25 / 30  # 25 have abstracts
    assert result["reason"] == "Quality gate passed"


@pytest.mark.unit
def test_insufficient_paper_count_fails(mock_paper_with_all_fields):
    """Test case 2: Insufficient paper count (<25) fails with appropriate reason."""
    papers = [mock_paper_with_all_fields] * 20  # Only 20 papers

    result = validate_quality(papers)

    assert result["passed"] is False
    assert result["metrics"]["paper_count"] == 20
    assert "paper count" in result["reason"].lower()


@pytest.mark.unit
def test_missing_fields_handled_gracefully(mock_paper_missing_abstract, mock_paper_with_all_fields):
    """Test case 5: Missing abstracts handled gracefully."""
    # Half have abstracts, half don't
    papers = [mock_paper_with_all_fields] * 15 + [mock_paper_missing_abstract] * 15  # 30 papers

    result = validate_quality(papers)

    assert result["passed"] is False  # Fails due to abstract coverage < threshold
    assert result["metrics"]["paper_count"] == 30
    assert result["metrics"]["abstract_coverage"] == 15 / 30  # Half have abstracts


@pytest.mark.unit
def test_invalid_papers_input_raises():
    """Test that invalid papers input raises appropriate errors."""
    # Not a list
    with pytest.raises(TypeError, match="papers must be a list"):
        validate_quality("not a list")  # type: ignore

    # Empty list
    with pytest.raises(ValueError, match="papers list cannot be empty"):
        validate_quality([])

    # Paper not a dict
    with pytest.raises(TypeError, match="paper at index 0 must be a dictionary"):
        validate_quality(["not a dict"])  # type: ignore

    # Missing paperId
    with pytest.raises(ValueError, match="missing required 'paperId' field"):
        validate_quality([{"title": "no paperId"}])


@pytest.mark.unit
def test_empty_abstract_handled():
    """Test that empty or whitespace-only abstracts are not counted."""
    papers = [
        {"paperId": "p1", "abstract": "valid abstract"},
        {"paperId": "p2", "abstract": ""},  # Empty string
        {"paperId": "p3", "abstract": "   "},  # Whitespace only
        {"paperId": "p4"},  # No abstract field
    ]

    result = validate_quality(papers)

    assert result["metrics"]["abstract_coverage"] == 1 / 4  # Only 1 valid abstract
