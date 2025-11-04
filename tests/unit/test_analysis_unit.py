import time
import pytest
from loguru import logger

from agents.analysis import AnalysisCredibilityScorer


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata, expected_tier, min_score",
    [
        (
            {
                "paper_id": "nature_2023",
                "citation_count": 250,
                "year": 2023,
                "journal": "Nature",
            },
            "high",
            75,
        ),
        (
            {
                "paper_id": "mid_2020",
                "citation_count": 45,
                "year": 2020,
                "journal": "Journal of Testing",
            },
            "medium",
            50,
        ),
        (
            {
                "paper_id": "preprint_2015",
                "citation_count": 3,
                "year": 2015,
                "journal": "arXiv",
            },
            "low",
            0,
        ),
    ],
)
def test_score_paper_parametrized(metadata, expected_tier, min_score):
    """CredibilityScorer should assign correct tiers and scores for representative inputs."""
    start = time.time()
    score = AnalysisCredibilityScorer.score_paper(metadata)
    elapsed = time.time() - start
    # fast unit - should be near-instant
    assert score.tier == expected_tier
    assert score.score >= min_score
    assert isinstance(score.rationale, str)
    # performance smoke: each call < 0.1s
    assert elapsed < 0.1


@pytest.mark.unit
def test_score_paper_handles_missing_and_bad_types():
    # Missing fields -> default medium
    score = AnalysisCredibilityScorer.score_paper({"paper_id": "x"})
    assert score.tier == "medium"
    assert score.score == 50

    # Non-dict input -> default medium
    score2 = AnalysisCredibilityScorer.score_paper(None)
    assert score2.tier == "medium"
    assert score2.score == 50
