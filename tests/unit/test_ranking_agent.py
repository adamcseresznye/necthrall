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
        "embedding": {"specter_v2": emb},  # High similarity
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
        "embedding": {"specter_v2": emb},  # Less similar
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
        "embedding": {"specter_v2": emb},  # Moderate similarity
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
        "embedding": {"specter_v2": emb},
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
        "embedding": {"specter_v2": emb},
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

    input_data = {"papers": papers, "query_embedding": sample_query_embedding}
    finalists = agent.rank_papers(input_data)

    # Should return all 3 papers since < 10 total
    assert len(finalists) == 3

    # Check that all have score fields
    for paper in finalists:
        assert "composite_score" in paper
        assert "semantic_score" in paper
        assert "authority_score" in paper
        assert "impact_score" in paper
        assert "recency_score" in paper
        assert 0.0 <= paper["composite_score"] <= 1.0

    # High similarity paper should rank highest (0.4 weight)
    assert finalists[0]["paperId"] == "paper_high_sim"
    # Recent paper should rank above high citations (due to recency boost)
    assert finalists[1]["paperId"] == "paper_recent"
    assert finalists[2]["paperId"] == "paper_high_cit"


@pytest.mark.unit
def test_high_semantic_similarity_ranks_higher_than_low_similarity_high_citations(
    sample_query_embedding,
):
    """Test case 2: High semantic similarity paper ranks higher than low similarity but high citations."""
    agent = RankingAgent()

    # Create two papers: one with perfect similarity but low citations, one with low similarity but high citations
    paper_sim = {
        "paperId": "perfect_sim",
        "embedding": {"specter_v2": sample_query_embedding},  # Perfect similarity
        "influentialCitationCount": 1,
        "citationCount": 5,
        "year": 2020,
    }

    paper_cit = {
        "paperId": "high_cit",
        "embedding": {
            "specter_v2": -sample_query_embedding
        },  # Opposite similarity (~ -1.0)
        "influentialCitationCount": 100,
        "citationCount": 500,
        "year": 2020,
    }

    papers = [paper_sim, paper_cit]
    input_data = {"papers": papers, "query_embedding": sample_query_embedding}
    finalists = agent.rank_papers(input_data)

    # Perfect similarity should rank higher despite low citations
    assert finalists[0]["paperId"] == "perfect_sim"
    assert finalists[1]["paperId"] == "high_cit"

    # Verify scores
    assert finalists[0]["semantic_score"] > finalists[1]["semantic_score"]
    assert finalists[0]["composite_score"] > finalists[1]["composite_score"]


@pytest.mark.unit
def test_recent_paper_gets_recency_boost(sample_query_embedding):
    """Test case 3: Recent paper (2024) gets recency boost over older paper (2015)."""
    agent = RankingAgent()

    paper_recent = {
        "paperId": "recent",
        "embedding": {"specter_v2": sample_query_embedding},
        "influentialCitationCount": 10,
        "citationCount": 50,
        "year": 2024,  # Max recency
    }

    paper_old = {
        "paperId": "old",
        "embedding": {"specter_v2": sample_query_embedding},
        "influentialCitationCount": 10,
        "citationCount": 50,
        "year": 2015,  # Min recency
    }

    papers = [paper_recent, paper_old]
    input_data = {"papers": papers, "query_embedding": sample_query_embedding}
    finalists = agent.rank_papers(input_data)

    # Recent paper should rank higher
    assert finalists[0]["paperId"] == "recent"
    assert finalists[1]["paperId"] == "old"

    # Recency scores should differ significantly
    assert finalists[0]["recency_score"] == 0.9  # 2024 -> 0.9
    assert finalists[1]["recency_score"] == 0.0  # 2015 -> 0.0


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

    input_data = {"papers": papers, "query_embedding": sample_query_embedding}
    finalists = agent.rank_papers(input_data)

    # Should return all papers
    assert len(finalists) == 3

    # Check specific handling
    for paper in finalists:
        if paper["paperId"] == "paper_no_emb":
            assert paper["semantic_score"] == 0.0  # No embedding -> similarity = 0
        elif paper["paperId"] == "paper_no_cit":
            assert paper["authority_score"] == 0.0  # No influential citations -> 0
            assert paper["impact_score"] == 0.0  # No total citations -> 0
        elif paper["paperId"] == "paper_no_year":
            assert paper["recency_score"] == 0.0  # No year -> min recency = 0


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
                "embedding": {"specter_v2": sample_query_embedding},
                "influentialCitationCount": 10,
                "citationCount": 50,
                "year": 2020,
            }
        )

    input_data = {"papers": papers, "query_embedding": sample_query_embedding}
    finalists = agent.rank_papers(input_data)

    # Should return all 5 papers
    assert len(finalists) == 5

    # All should have identical scores
    scores = [p["composite_score"] for p in finalists]
    assert all(s == scores[0] for s in scores)

    # Order should be stable (same as input order since scores equal)
    assert [p["paperId"] for p in finalists] == [f"identical_{i}" for i in range(5)]


@pytest.mark.unit
def test_invalid_inputs_raise_errors():
    """Test that invalid inputs raise appropriate errors."""
    agent = RankingAgent()

    # Invalid papers
    with pytest.raises(ValueError, match="papers must be a list"):
        agent.rank_papers(
            {"papers": "not a list", "query_embedding": np.random.rand(384)}
        )

    # Invalid query_embedding
    with pytest.raises(ValueError, match="query_embedding must be a numpy array"):
        agent.rank_papers({"papers": [], "query_embedding": "not an array"})

    # Wrong shape
    with pytest.raises(
        ValueError, match="query_embedding must be 384 or 768-dimensional"
    ):
        agent.rank_papers({"papers": [], "query_embedding": np.random.rand(100)})

    # Non-finite values
    with pytest.raises(ValueError, match="query_embedding contains non-finite values"):
        agent.rank_papers(
            {"papers": [], "query_embedding": np.array([1.0] * 383 + [float("nan")])}
        )


@pytest.mark.unit
def test_empty_papers_list():
    """Test handling of empty papers list."""
    agent = RankingAgent()
    input_data = {"papers": [], "query_embedding": np.random.rand(384)}

    finalists = agent.rank_papers(input_data)
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
                "embedding": {"specter_v2": sample_query_embedding - similarity_offset},
                "influentialCitationCount": 10,
                "citationCount": 50,
                "year": 2020,
            }
        )

    input_data = {"papers": papers, "query_embedding": sample_query_embedding}
    finalists = agent.rank_papers(input_data)

    # Should return exactly 10 papers
    assert len(finalists) == 10

    # Should be the top 10 (highest similarity first)
    returned_ids = [p["paperId"] for p in finalists]
    expected_ids = [f"paper_{i}" for i in range(10)]
    assert returned_ids == expected_ids


@pytest.mark.unit
def test_log_normalization_handles_zero_citations():
    """Test that log normalization works when all papers have zero citations."""
    agent = RankingAgent()

    papers = [
        {
            "paperId": "zero_auth",
            "embedding": {"specter_v2": np.random.rand(384)},
            "influentialCitationCount": 0,
            "citationCount": 100,
            "year": 2020,
        },
        {
            "paperId": "zero_impact",
            "embedding": {"specter_v2": np.random.rand(384)},
            "influentialCitationCount": 50,
            "citationCount": 0,
            "year": 2020,
        },
    ]

    input_data = {"papers": papers, "query_embedding": np.random.rand(384)}
    finalists = agent.rank_papers(input_data)

    # Authority score should be 0 for zero influential citations
    assert (
        finalists[0]["authority_score"] == 0.0 or finalists[1]["authority_score"] == 0.0
    )
    # Impact score should be 0 for zero total citations
    assert finalists[0]["impact_score"] == 0.0 or finalists[1]["impact_score"] == 0.0


@pytest.mark.unit
def test_year_clamping():
    """Test that years outside 2015-2025 are clamped."""
    agent = RankingAgent()

    papers = [
        {
            "paperId": "too_old",
            "embedding": {"specter_v2": np.random.rand(384)},
            "influentialCitationCount": 10,
            "citationCount": 50,
            "year": 2000,  # Below 2015
        },
        {
            "paperId": "too_new",
            "embedding": {"specter_v2": np.random.rand(384)},
            "influentialCitationCount": 10,
            "citationCount": 50,
            "year": 2030,  # Above 2025
        },
    ]

    input_data = {"papers": papers, "query_embedding": np.random.rand(384)}
    finalists = agent.rank_papers(input_data)

    # Both should have recency score clamped to valid range
    recency_scores = [paper["recency_score"] for paper in finalists]
    assert 0.0 in recency_scores  # 2000 clamped to 2015 -> 0.0
    assert 1.0 in recency_scores  # 2030 clamped to 2025 -> 1.0


@pytest.mark.unit
def test_composite_score_calculation():
    """Test composite score calculation with known inputs."""
    agent = RankingAgent()

    # Test with specific sub-scores
    semantic = np.array([1.0, 0.5, 0.0])
    authority = np.array([0.8, 0.6, 0.2])
    impact = np.array([0.9, 0.4, 0.1])
    recency = np.array([0.7, 0.8, 0.9])

    composite = agent._compute_composite_scores(semantic, authority, impact, recency)

    expected = np.array(
        [
            0.4 * 1.0
            + 0.3 * 0.8
            + 0.2 * 0.9
            + 0.1 * 0.7,  # 0.4 + 0.24 + 0.18 + 0.07 = 0.89
            0.4 * 0.5
            + 0.3 * 0.6
            + 0.2 * 0.4
            + 0.1 * 0.8,  # 0.2 + 0.18 + 0.08 + 0.08 = 0.54
            0.4 * 0.0
            + 0.3 * 0.2
            + 0.2 * 0.1
            + 0.1 * 0.9,  # 0.0 + 0.06 + 0.02 + 0.09 = 0.17
        ]
    )

    np.testing.assert_array_almost_equal(composite, expected)


@pytest.mark.unit
def test_composite_score_ranking_behavior():
    """Test that composite scores rank papers correctly."""
    agent = RankingAgent()

    # Create papers with different score profiles
    papers = [
        {
            "paperId": "high_semantic",
            "title": "High Semantic",
            "embedding": {"specter_v2": np.ones(384)},
            "influentialCitationCount": 10,
            "citationCount": 20,
            "year": 2020,
        },
        {
            "paperId": "high_citations",
            "title": "High Citations",
            "embedding": {"specter_v2": np.zeros(384)},
            "influentialCitationCount": 1000,
            "citationCount": 5000,
            "year": 2015,
        },
        {
            "paperId": "high_recency",
            "title": "High Recency",
            "embedding": {"specter_v2": np.zeros(384)},
            "influentialCitationCount": 10,
            "citationCount": 20,
            "year": 2025,
        },
    ]

    input_data = {"papers": papers, "query_embedding": np.ones(384)}
    finalists = agent.rank_papers(input_data)

    # High semantic should rank first (perfect similarity)
    assert finalists[0]["paperId"] == "high_semantic"
    # High citations should rank second
    assert finalists[1]["paperId"] == "high_citations"
    # High recency should rank third
    assert finalists[2]["paperId"] == "high_recency"


@pytest.mark.unit
def test_composite_score_recency_boost():
    """Test that recency provides boost in composite scoring."""
    agent = RankingAgent()

    # Two papers identical except for year
    base_paper = {
        "embedding": {"specter_v2": np.random.rand(384)},
        "influentialCitationCount": 50,
        "citationCount": 100,
    }

    papers = [
        {**base_paper, "paperId": "old", "title": "Old Paper", "year": 2015},
        {**base_paper, "paperId": "new", "title": "New Paper", "year": 2025},
    ]

    input_data = {"papers": papers, "query_embedding": np.random.rand(384)}
    finalists = agent.rank_papers(input_data)

    # New paper should rank higher due to recency boost
    assert finalists[0]["paperId"] == "new"
    assert finalists[1]["paperId"] == "old"

    # Recency scores should differ by 1.0
    assert abs(finalists[0]["recency_score"] - finalists[1]["recency_score"]) == 1.0


@pytest.mark.unit
def test_composite_score_edge_cases():
    """Test composite score calculation with edge cases."""
    agent = RankingAgent()

    # Test with all zeros
    zeros = np.zeros(3)
    composite = agent._compute_composite_scores(zeros, zeros, zeros, zeros)
    np.testing.assert_array_equal(composite, np.zeros(3))

    # Test with all ones
    ones = np.ones(3)
    composite = agent._compute_composite_scores(ones, ones, ones, ones)
    np.testing.assert_array_almost_equal(composite, np.ones(3))

    # Test with mixed values
    semantic = np.array([0.0, 0.5, 1.0])
    authority = np.array([1.0, 0.5, 0.0])
    impact = np.array([0.5, 1.0, 0.5])
    recency = np.array([0.2, 0.8, 0.6])

    composite = agent._compute_composite_scores(semantic, authority, impact, recency)
    expected = 0.4 * semantic + 0.3 * authority + 0.2 * impact + 0.1 * recency
    np.testing.assert_array_almost_equal(composite, expected)
