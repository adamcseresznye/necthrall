import pytest
import time

pytestmark = [pytest.mark.unit]

from agents.analysis import AnalysisCredibilityScorer
from models.state import CredibilityScore


@pytest.mark.parametrize(
    "metadata, expected",
    [
        # zero citations -> low tier expected (low citations + moderately recent)
        (
            {
                "paper_id": "zero",
                "citation_count": 0,
                "year": 2020,
                "journal": "Some Journal",
            },
            {"tier": "low"},
        ),
        # future publication year (clamped to CURRENT_YEAR) -> ensure handled and recency uses CURRENT_YEAR
        (
            {
                "paper_id": "future",
                "citation_count": 10,
                "year": AnalysisCredibilityScorer.CURRENT_YEAR + 5,
                "journal": "Journal X",
            },
            {"tier_in": ("low", "medium", "high")},
        ),
        # special journal variation should detect top-tier (substring match)
        (
            {
                "paper_id": "nature_comm",
                "citation_count": 120,
                "year": 2023,
                "journal": "Nature Communications",
            },
            {"expect_venue_label": "top-tier"},
        ),
        # complete metadata absence -> default medium
        (
            {},
            {"tier": "medium", "score": 50},
        ),
        # None metadata (non-dict) -> default medium
        (None, {"tier": "medium", "score": 50}),
    ],
)
def test_edge_cases_parametrized(metadata, expected):
    score = AnalysisCredibilityScorer.score_paper(metadata)
    assert isinstance(score, CredibilityScore)

    if "tier" in expected:
        assert score.tier == expected["tier"]

    if "score" in expected:
        assert score.score == expected["score"]

    if "tier_in" in expected:
        assert score.tier in expected["tier_in"]

    if "expect_venue_label" in expected:
        # rationale should contain 'top-tier' substring
        assert "top-tier" in score.rationale or "top-tier" in score.rationale.lower()


@pytest.mark.performance
def test_scoring_performance_under_1ms():
    """Test that scoring achieves <1ms per paper on average for bulk operations."""
    # Generate a list of diverse metadata to score
    test_papers = [
        {
            "paper_id": f"paper_{i}",
            "citation_count": i * 10,
            "year": 2020 + (i % 5),
            "journal": "Nature" if i % 10 == 0 else "Some Journal",
        }
        for i in range(1000)
    ]

    # Time the bulk scoring
    start_time = time.perf_counter()
    scores = AnalysisCredibilityScorer.score_papers(test_papers)
    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000
    avg_ms_per_paper = elapsed_ms / len(test_papers)

    # Assert <1ms per paper
    assert (
        avg_ms_per_paper < 1.0
    ), f"Average scoring time {avg_ms_per_paper:.3f}ms per paper exceeds 1ms target"

    # Ensure all scores are valid
    assert len(scores) == len(test_papers)
    for score in scores:
        assert isinstance(score, CredibilityScore)
        assert 0 <= score.score <= 100
        assert score.tier in ("low", "medium", "high")
