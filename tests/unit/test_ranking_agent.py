"""Unit tests for RankingAgent using BM25/TF-IDF/LSA ranking.

The ranking now uses 3-signal RRF (BM25, TF-IDF, LSA) based on text matching,
plus authority (citations) and recency boosts.
"""

import pytest

from agents.ranking_agent import RankingAgent
from models.state import Paper


@pytest.fixture
def papers_with_keyword_matches():
    """Papers with different keyword matching potential for 'machine learning'."""
    return [
        Paper(
            paperId="ml_paper",
            title="Machine Learning for Drug Discovery",
            abstract="We apply machine learning techniques to discover new drugs.",
            influentialCitationCount=50,
            citationCount=200,
            year=2023,
        ),
        Paper(
            paperId="unrelated_paper",
            title="Climate Change in Arctic",
            abstract="Study of temperature changes in the Arctic region.",
            influentialCitationCount=100,
            citationCount=500,
            year=2022,
        ),
        Paper(
            paperId="another_ml_paper",
            title="Deep Learning Methods",
            abstract="Neural networks and deep learning for image recognition.",
            influentialCitationCount=30,
            citationCount=150,
            year=2024,
        ),
    ]


@pytest.fixture
def papers_with_citations():
    """Papers with varying citation counts."""
    return [
        Paper(
            paperId="high_cit",
            title="Research Paper One",
            abstract="Abstract about research",
            influentialCitationCount=500,
            citationCount=2000,
            year=2020,
        ),
        Paper(
            paperId="low_cit",
            title="Research Paper Two",
            abstract="Abstract about research",
            influentialCitationCount=5,
            citationCount=20,
            year=2020,
        ),
    ]


@pytest.fixture
def papers_with_varying_recency():
    """Papers with varying years."""
    return [
        Paper(
            paperId="old_paper",
            title="Research on Methods",
            abstract="An old study on research methods.",
            influentialCitationCount=50,
            citationCount=200,
            year=2015,
        ),
        Paper(
            paperId="new_paper",
            title="Research on Methods",
            abstract="A new study on research methods.",
            influentialCitationCount=50,
            citationCount=200,
            year=2024,
        ),
    ]


@pytest.mark.unit
def test_ranking_agent_returns_all_papers_under_limit(papers_with_keyword_matches):
    """RankingAgent returns all papers when under the limit."""
    agent = RankingAgent()
    finalists = agent.rank_papers(papers_with_keyword_matches, query="machine learning")
    assert len(finalists) == 3


@pytest.mark.unit
def test_ranking_produces_score_attributes(papers_with_keyword_matches):
    """Ranked papers have the expected score attributes."""
    agent = RankingAgent()
    finalists = agent.rank_papers(papers_with_keyword_matches, query="machine learning")

    for paper in finalists:
        assert hasattr(paper, "final_score")
        assert hasattr(paper, "relevance_score")
        assert hasattr(paper, "authority_score")
        assert hasattr(paper, "recency_score")


@pytest.mark.unit
def test_keyword_matching_improves_relevance():
    """Papers with query keywords in title/abstract should have higher relevance."""
    agent = RankingAgent()

    papers = [
        Paper(
            paperId="matching",
            title="Study on Fasting Effects",
            abstract="Intermittent fasting improves metabolism.",
            citationCount=100,
            year=2022,
        ),
        Paper(
            paperId="non_matching",
            title="Study on Climate Change",
            abstract="Temperature readings from Antarctic stations.",
            citationCount=100,
            year=2022,
        ),
    ]

    finalists = agent.rank_papers(papers, query="fasting metabolism")

    # The paper with matching keywords should rank higher
    assert finalists[0].paperId == "matching"


@pytest.mark.unit
def test_recency_boost_affects_ranking(papers_with_varying_recency):
    """More recent papers get a recency boost."""
    agent = RankingAgent()
    finalists = agent.rank_papers(papers_with_varying_recency, query="research methods")

    # New paper should have higher recency score
    new_paper = next(p for p in finalists if p.paperId == "new_paper")
    old_paper = next(p for p in finalists if p.paperId == "old_paper")
    assert new_paper.recency_score > old_paper.recency_score


@pytest.mark.unit
def test_authority_score_reflects_citations(papers_with_citations):
    """Papers with more citations should have higher authority scores."""
    agent = RankingAgent()
    finalists = agent.rank_papers(papers_with_citations, query="research")

    high_cit_paper = next(p for p in finalists if p.paperId == "high_cit")
    low_cit_paper = next(p for p in finalists if p.paperId == "low_cit")
    assert high_cit_paper.authority_score > low_cit_paper.authority_score


@pytest.mark.unit
def test_empty_papers_list_returns_empty():
    """Empty input returns empty list."""
    agent = RankingAgent()
    finalists = agent.rank_papers([], query="test")
    assert finalists == []


@pytest.mark.unit
def test_papers_missing_fields_handled():
    """Papers with missing optional fields don't crash ranking."""
    agent = RankingAgent()

    papers = [
        Paper(
            paperId="minimal",
            title="Minimal Paper",
            abstract=None,  # Missing abstract
            citationCount=0,  # Use 0 instead of None
            year=2020,
        ),
    ]

    finalists = agent.rank_papers(papers, query="test")
    assert len(finalists) == 1
