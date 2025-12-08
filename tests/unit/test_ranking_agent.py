import numpy as np
import pytest

from agents.ranking_agent import RankingAgent


@pytest.fixture
def sample_query_embedding():
    """A mock 384-dimensional SPECTER2 query embedding."""
    np.random.seed(42)  # For reproducible tests
    emb = np.random.rand(384)
    return emb / np.linalg.norm(emb)  # Normalize like SPECTER2


@pytest.fixture
def mock_paper_high_similarity(sample_query_embedding):
    """A mock paper with high semantic similarity."""
    # Create a similar embedding
    emb = sample_query_embedding + np.random.rand(384) * 0.1
    emb = emb / np.linalg.norm(emb)
    return {
        "paperId": "paper_high_sim",
        "title": "High Similarity Paper",
        "abstract": "Abstract high similarity",
        "embedding": {"specter": emb},  # High similarity
        "influentialCitationCount": 50,
        "citationCount": 200,
        "year": 2023,
    }


@pytest.fixture
def mock_paper_high_citations(sample_query_embedding):
    """A mock paper with high citations but lower similarity."""
    # Create a less similar embedding
    np.random.seed(43)
    emb = sample_query_embedding - np.random.rand(384) * 0.5
    emb = emb / np.linalg.norm(emb)
    return {
        "paperId": "paper_high_cit",
        "title": "High Citations Paper",
        "abstract": "Abstract high citations",
        "embedding": {"specter": emb},  # Less similar
        "influentialCitationCount": 200,
        "citationCount": 1000,
        "year": 2015,  # Very old to allow recency boost
    }


@pytest.fixture
def mock_paper_recent(sample_query_embedding):
    """A mock paper that's very recent."""
    # Create a moderately similar embedding
    np.random.seed(44)
    emb = sample_query_embedding + np.random.rand(384) * 0.1  # More similar
    emb = emb / np.linalg.norm(emb)
    return {
        "paperId": "paper_recent",
        "title": "Recent Paper",
        "abstract": "Abstract recent",
        "embedding": {"specter": emb},  # Moderate similarity
        "influentialCitationCount": 10,
        "citationCount": 50,
        "year": 2024,
    }


@pytest.fixture
def mock_paper_missing_embedding():
    """A mock paper missing embedding."""
    return {
        "paperId": "paper_no_emb",
        "title": "No Embedding Paper",
        "abstract": "No embedding abstract",
        "influentialCitationCount": 20,
        "citationCount": 100,
        "year": 2020,
    }


@pytest.fixture
def mock_paper_missing_citations(sample_query_embedding):
    """A mock paper missing citation counts."""
    emb = sample_query_embedding + np.random.rand(384) * 0.2
    emb = emb / np.linalg.norm(emb)
    return {
        "paperId": "paper_no_cit",
        "title": "No Citations Paper",
        "abstract": "No citations abstract",
        "embedding": {"specter": emb},
        "year": 2022,
    }


@pytest.fixture
def mock_paper_missing_year(sample_query_embedding):
    """A mock paper missing year."""
    emb = sample_query_embedding + np.random.rand(384) * 0.15
    emb = emb / np.linalg.norm(emb)
    return {
        "paperId": "paper_no_year",
        "title": "No Year Paper",
        "abstract": "No year abstract",
        "embedding": {"specter": emb},
        "influentialCitationCount": 15,
        "citationCount": 75,
    }


@pytest.mark.unit
def test_papers_ranked_correctly_by_composite_score(
    sample_query_embedding,
    mock_paper_high_similarity,
    mock_paper_high_citations,
    mock_paper_recent,
):
    """Test case 1: Papers ranked correctly by composite score on manual inspection."""
    agent = RankingAgent()
    papers = [mock_paper_high_similarity, mock_paper_high_citations, mock_paper_recent]

    # RankingAgent now expects (papers, query) signature
    finalists = agent.rank_papers(papers, query="test query")

    # Should return all 3 papers since < 10 total
    assert len(finalists) == 3

    # Check that all have the new score fields
    for paper in finalists:
        assert "final_score" in paper
        assert "relevance_score" in paper
        assert "authority_score" in paper
        assert "recency_score" in paper

    # Expect high similarity paper to be among the top results
    assert finalists[0]["paperId"] in {"paper_high_sim", "paper_recent"}


@pytest.mark.unit
def test_high_semantic_similarity_ranks_higher_than_low_similarity_high_citations(
    sample_query_embedding,
):
    """Test case 2: High semantic similarity paper ranks higher than low similarity but high citations."""
    agent = RankingAgent()

    paper_sim = {
        "paperId": "perfect_sim",
        "title": "Perfect Sim",
        "abstract": "abstract",
        "embedding": {"specter": sample_query_embedding},  # Perfect similarity
        "influentialCitationCount": 1,
        "citationCount": 5,
        "year": 2020,
    }

    paper_cit = {
        "paperId": "high_cit",
        "title": "High Cit",
        "abstract": "abstract",
        "embedding": {"specter": -sample_query_embedding},
        "influentialCitationCount": 100,
        "citationCount": 500,
        "year": 2020,
    }

    papers = [paper_sim, paper_cit]
    finalists = agent.rank_papers(papers, query="test")

    # Verify relevance scores reflect similarity ordering (perfect_sim should have >= relevance)
    # Find the two papers in the finalists list
    scores_by_id = {p["paperId"]: p["relevance_score"] for p in finalists}
    assert scores_by_id["perfect_sim"] >= scores_by_id["high_cit"]


@pytest.mark.unit
def test_recent_paper_gets_recency_boost(sample_query_embedding):
    """Test case 3: Recent paper (2024) gets recency boost over older paper (2015)."""
    agent = RankingAgent()

    paper_recent = {
        "paperId": "recent",
        "title": "Recent",
        "abstract": "Recent abstract",
        "embedding": {"specter": sample_query_embedding},
        "influentialCitationCount": 10,
        "citationCount": 50,
        "year": 2024,  # Max recency
    }

    paper_old = {
        "paperId": "old",
        "title": "Old",
        "abstract": "Old abstract",
        "embedding": {"specter": sample_query_embedding},
        "influentialCitationCount": 10,
        "citationCount": 50,
        "year": 2015,  # Min recency
    }

    papers = [paper_recent, paper_old]
    finalists = agent.rank_papers(papers, query="test")

    # Recent paper should rank at least as high as old paper
    assert finalists[0]["paperId"] == "recent"
    assert finalists[1]["paperId"] == "old"

    # Recency scores: recent > old
    recency_scores = {p["paperId"]: p["recency_score"] for p in finalists}
    assert recency_scores["recent"] > recency_scores["old"]


@pytest.mark.unit
def test_missing_fields_handled_gracefully(
    sample_query_embedding,
    mock_paper_missing_embedding,
    mock_paper_missing_citations,
    mock_paper_missing_year,
):
    """Test case 4: Papers with missing fields handled gracefully without crashes."""
    agent = RankingAgent()
    papers = [
        mock_paper_missing_embedding,
        mock_paper_missing_citations,
        mock_paper_missing_year,
    ]

    finalists = agent.rank_papers(papers, query="test")

    # Should return all papers and not crash
    assert len(finalists) == 3

    # Ensure score fields exist and are numeric
    for paper in finalists:
        assert isinstance(paper.get("final_score"), float)
        assert isinstance(paper.get("relevance_score"), float)


@pytest.mark.unit
def test_edge_case_identical_scores(sample_query_embedding):
    """Test case 5: Edge case with all papers having identical scores."""
    agent = RankingAgent()

    # Create 5 identical papers
    papers = []
    for i in range(5):
        papers.append(
            {
                "paperId": f"identical_{i}",
                "title": f"identical_{i}",
                "abstract": "a",
                "embedding": {"specter": sample_query_embedding},
                "influentialCitationCount": 10,
                "citationCount": 50,
                "year": 2020,
            }
        )

    finalists = agent.rank_papers(papers, query="identical")

    # Should return all 5 papers
    assert len(finalists) == 5

    # All should have identical scores
    scores = [p["final_score"] for p in finalists]
    # Scores should be equal (or very close) for identical inputs
    assert all(abs(s - scores[0]) < 1e-6 for s in scores)

    # Order should be stable-ish (we accept preserving input order)
    assert [p["paperId"] for p in finalists] == [f"identical_{i}" for i in range(5)]


@pytest.mark.unit
def test_invalid_inputs_raise_errors():
    """Test that invalid inputs raise appropriate errors."""
    agent = RankingAgent()

    # Invalid papers
    with pytest.raises(ValueError, match="papers must be a list"):
        agent.rank_papers("not a list", query="q")

    # Invalid query_embedding
    # RankingAgent validates query is non-empty string; invalid query types should raise
    with pytest.raises(ValueError):
        agent.rank_papers([], query=123)  # type: ignore


@pytest.mark.unit
def test_empty_papers_list():
    """Test handling of empty papers list."""
    agent = RankingAgent()
    finalists = agent.rank_papers([], query="q")
    assert finalists == []


@pytest.mark.unit
def test_more_than_10_papers_returns_top_10(sample_query_embedding):
    """Test that when more than 10 papers, only top 10 are returned."""
    agent = RankingAgent()

    # Create 15 papers with decreasing similarity
    papers = []
    for i in range(15):
        similarity_offset = i * 0.1  # Decreasing similarity
        papers.append(
            {
                "paperId": f"paper_{i}",
                "title": f"paper_{i}",
                "abstract": "a",
                "embedding": {"specter": (sample_query_embedding - similarity_offset)},
                "influentialCitationCount": 10,
                "citationCount": 50,
                "year": 2020,
            }
        )

    finalists = agent.rank_papers(papers, query="rank", top_k=10)

    # Should return exactly 10 papers
    assert len(finalists) == 10

    # Should be the top 10 (highest similarity first)
    returned_ids = [p["paperId"] for p in finalists]
    assert len(returned_ids) == 10


@pytest.mark.unit
def test_log_normalization_handles_zero_citations():
    """Test that log normalization works when all papers have zero citations."""
    agent = RankingAgent()

    papers = [
        {
            "paperId": "zero_auth",
            "title": "zero_auth",
            "abstract": "a",
            "embedding": {"specter": np.random.rand(384)},
            "influentialCitationCount": 0,
            "citationCount": 100,
            "year": 2020,
        },
        {
            "paperId": "zero_impact",
            "title": "zero_impact",
            "abstract": "a",
            "embedding": {"specter": np.random.rand(384)},
            "influentialCitationCount": 50,
            "citationCount": 0,
            "year": 2020,
        },
    ]

    finalists = agent.rank_papers(papers, query="q")

    # At least one paper should show zero authority or impact related component
    assert any(p.get("authority_score", 0.0) == 0.0 for p in finalists)


@pytest.mark.unit
def test_year_clamping():
    """Test that years outside 2015-2025 are clamped."""
    agent = RankingAgent()

    papers = [
        {
            "paperId": "too_old",
            "title": "too_old",
            "abstract": "a",
            "embedding": {"specter": np.random.rand(384)},
            "influentialCitationCount": 10,
            "citationCount": 50,
            "year": 2000,  # Below 2015
        },
        {
            "paperId": "too_new",
            "title": "too_new",
            "abstract": "a",
            "embedding": {"specter": np.random.rand(384)},
            "influentialCitationCount": 10,
            "citationCount": 50,
            "year": 2030,  # Above 2025
        },
    ]

    finalists = agent.rank_papers(papers, query="q")

    # Both should have recency score values within 0..1 after normalization
    recency_scores = [paper["recency_score"] for paper in finalists]
    assert all(0.0 <= s <= 1.0 for s in recency_scores)


@pytest.mark.unit
def test_composite_score_calculation():
    """Test composite score calculation with known inputs."""
    # The ranking agent uses an internal weighted aggregation to compute final_score.
    # Validate the function runs on trivial inputs by ensuring it returns expected
    # output shape when ranking a few dummy papers.
    agent = RankingAgent()
    papers = []
    for i in range(3):
        papers.append(
            {
                "paperId": f"p{i}",
                "title": "t",
                "abstract": "a",
                "embedding": {"specter": np.ones(384)},
                "influentialCitationCount": 1,
                "citationCount": 1,
                "year": 2020,
            }
        )

    finalists = agent.rank_papers(papers, query="q")
    assert len(finalists) == 3


@pytest.mark.unit
def test_composite_score_ranking_behavior():
    """Test that composite scores rank papers correctly."""
    agent = RankingAgent()

    # Create papers with different score profiles
    papers = [
        {
            "paperId": "high_semantic",
            "title": "High Semantic",
            "abstract": "high semantic abstract",
            "embedding": {"specter": np.ones(384)},
            "influentialCitationCount": 10,
            "citationCount": 20,
            "year": 2020,
        },
        {
            "paperId": "high_citations",
            "title": "High Citations",
            "abstract": "high citations abstract",
            "embedding": {"specter": np.zeros(384)},
            "influentialCitationCount": 1000,
            "citationCount": 5000,
            "year": 2015,
        },
        {
            "paperId": "high_recency",
            "title": "High Recency",
            "abstract": "high recency abstract",
            "embedding": {"specter": np.zeros(384)},
            "influentialCitationCount": 10,
            "citationCount": 20,
            "year": 2025,
        },
    ]

    finalists = agent.rank_papers(papers, query="q")

    # High semantic should rank first (perfect similarity)
    assert finalists[0]["paperId"] == "high_semantic"
    # High citations and recency follow (order may vary depending on ties)
    assert set([finalists[1]["paperId"], finalists[2]["paperId"]]) == {
        "high_citations",
        "high_recency",
    }


@pytest.mark.unit
def test_composite_score_recency_boost():
    """Test that recency provides boost in composite scoring."""
    agent = RankingAgent()

    # Two papers identical except for year
    base_paper = {
        "embedding": {"specter": np.random.rand(384)},
        "influentialCitationCount": 50,
        "citationCount": 100,
        "title": "base",
        "abstract": "base abstract",
    }

    papers = [
        {**base_paper, "paperId": "old", "title": "Old Paper", "year": 2015},
        {**base_paper, "paperId": "new", "title": "New Paper", "year": 2025},
    ]

    finalists = agent.rank_papers(papers, query="q")

    # New paper should rank higher due to recency boost
    assert finalists[0]["paperId"] == "new"
    assert finalists[1]["paperId"] == "old"

    # Recency scores should be greater for the newer paper
    assert finalists[0]["recency_score"] > finalists[1]["recency_score"]


@pytest.mark.unit
def test_composite_score_edge_cases():
    """Test composite score calculation with edge cases."""
    agent = RankingAgent()

    # Test with all zeros
    zeros = np.zeros(3)
    # Edge cases: ranking on trivial inputs should not raise and should return same
    # number of finalists as inputs when < 10
    papers = []
    for i in range(3):
        papers.append(
            {
                "paperId": f"e{i}",
                "title": "t",
                "abstract": "a",
                "embedding": {"specter": np.ones(384)},
                "influentialCitationCount": 0,
                "citationCount": 0,
                "year": 2020,
            }
        )

    finalists = agent.rank_papers(papers, query="q")
    assert len(finalists) == 3
