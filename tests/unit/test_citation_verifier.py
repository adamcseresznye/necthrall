"""Unit tests for CitationVerifier.

Tests citation parsing and validation logic for synthesized answers.
"""

import pytest

from utils.citation_verifier import CitationVerifier


@pytest.fixture
def verifier():
    """Return a fresh CitationVerifier instance."""
    return CitationVerifier()


@pytest.fixture
def sample_passages():
    """Return a sample list of 3 passages."""
    return [
        {"content": "Passage 1 content", "paper_id": "paper_1"},
        {"content": "Passage 2 content", "paper_id": "paper_2"},
        {"content": "Passage 3 content", "paper_id": "paper_3"},
    ]


class TestCitationVerifier:
    """Test suite for CitationVerifier.verify() method."""

    @pytest.mark.unit
    def test_valid_citations(self, verifier, sample_passages):
        """Test case 1: Valid citations within range are accepted."""
        answer = "This claim is supported [1] and confirmed by another study [2]."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == [1, 2]
        assert result["invalid_citations"] == []
        assert "valid" in result["reason"].lower()

    @pytest.mark.unit
    def test_all_valid_citations_single(self, verifier, sample_passages):
        """Single valid citation should be accepted."""
        answer = "The evidence suggests [3] this is correct."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == [3]
        assert result["invalid_citations"] == []

    @pytest.mark.unit
    def test_invalid_citation_exceeds_passage_count(self, verifier, sample_passages):
        """Test case 2: Citation exceeding passage count is invalid."""
        answer = "This is claimed [1] but also [5] which is wrong."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is False
        assert result["citations_found"] == [1, 5]
        assert result["invalid_citations"] == [5]
        assert "5" in result["reason"]

    @pytest.mark.unit
    def test_invalid_citation_zero(self, verifier, sample_passages):
        """Citation [0] is invalid (indices start at 1)."""
        answer = "Invalid zero citation [0] here."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is False
        assert result["citations_found"] == [0]
        assert result["invalid_citations"] == [0]

    @pytest.mark.unit
    def test_no_citations_found(self, verifier, sample_passages):
        """Test case 3: No citations in text is technically valid."""
        answer = "This is a plain answer without any citations."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == []
        assert result["invalid_citations"] == []
        assert "no citations" in result["reason"].lower()

    @pytest.mark.unit
    def test_malformed_tags_ignored(self, verifier, sample_passages):
        """Test case 4: Malformed tags like (1), [A], {1} are ignored."""
        answer = "This has (1) parentheses and [A] letters and {1} braces."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == []
        assert result["invalid_citations"] == []

    @pytest.mark.unit
    def test_mixed_valid_and_malformed(self, verifier, sample_passages):
        """Valid citations are extracted even with malformed tags present."""
        answer = "Valid [1] citation mixed with (2) and [B] malformed ones."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == [1]
        assert result["invalid_citations"] == []

    @pytest.mark.unit
    def test_duplicate_citations_counted_once(self, verifier, sample_passages):
        """Duplicate citations should be de-duplicated in output."""
        answer = "Cited [1] here and [1] again and [2] once."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == [1, 2]
        assert result["invalid_citations"] == []

    @pytest.mark.unit
    def test_multiple_invalid_citations(self, verifier, sample_passages):
        """Multiple invalid citations are all reported."""
        answer = "Invalid [4] and [10] and [0] citations."
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is False
        assert result["citations_found"] == [0, 4, 10]
        assert result["invalid_citations"] == [0, 4, 10]

    @pytest.mark.unit
    def test_empty_passages_list(self, verifier):
        """With empty passages, any citation is invalid."""
        answer = "This cites [1] but there are no passages."
        result = verifier.verify(answer, [])

        assert result["valid"] is False
        assert result["citations_found"] == [1]
        assert result["invalid_citations"] == [1]

    @pytest.mark.unit
    def test_empty_answer(self, verifier, sample_passages):
        """Empty answer has no citations, which is valid."""
        answer = ""
        result = verifier.verify(answer, sample_passages)

        assert result["valid"] is True
        assert result["citations_found"] == []
        assert result["invalid_citations"] == []

    @pytest.mark.unit
    def test_large_citation_numbers(self, verifier):
        """Large citation numbers are handled correctly."""
        passages = [{"content": f"Passage {i}"} for i in range(100)]
        answer = "References [1], [50], [100] are valid, but [101] is not."
        result = verifier.verify(answer, passages)

        assert result["valid"] is False
        assert result["citations_found"] == [1, 50, 100, 101]
        assert result["invalid_citations"] == [101]

    @pytest.mark.unit
    def test_citations_sorted_in_output(self, verifier, sample_passages):
        """Citations in output should be sorted numerically."""
        answer = "Out of order [3] then [1] then [2]."
        result = verifier.verify(answer, sample_passages)

        assert result["citations_found"] == [1, 2, 3]

    @pytest.mark.unit
    def test_result_structure(self, verifier, sample_passages):
        """Verify the result dictionary has all required keys."""
        answer = "Some text [1]."
        result = verifier.verify(answer, sample_passages)

        assert "valid" in result
        assert "citations_found" in result
        assert "invalid_citations" in result
        assert "reason" in result
        assert isinstance(result["valid"], bool)
        assert isinstance(result["citations_found"], list)
        assert isinstance(result["invalid_citations"], list)
        assert isinstance(result["reason"], str)
