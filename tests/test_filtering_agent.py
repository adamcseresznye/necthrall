import pytest
from unittest.mock import Mock, patch
import numpy as np
from agents.filtering_agent import FilteringAgent
from models.state import State, Paper


@pytest.fixture
def mock_request():
    """Mock FastAPI Request with cached embedding model"""
    request = Mock()

    # Mock embedding model
    mock_model = Mock()
    mock_model.encode = Mock(
        side_effect=lambda texts, **kwargs: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        )
    )

    request.app.state.embedding_model = mock_model
    return request


@pytest.fixture
def agent(mock_request):
    return FilteringAgent(mock_request)


@pytest.fixture
def sample_papers():
    """Create 100 sample papers with varying metadata"""
    papers = []
    for i in range(100):
        papers.append(
            Paper(
                paper_id=f"openalex:{i}",
                title=f"CRISPR gene editing paper {i}",
                abstract=f"This paper discusses CRISPR-Cas9 mechanisms in detail {i}",
                authors=["Author"],
                year=2020 + (i % 5),  # Years 2020-2024
                journal="Nature",
                citation_count=i * 10,  # 0-990 citations
                doi=f"10.1000/{i}",
                pdf_url="https://example.com/paper.pdf",
                type="review" if i % 10 == 0 else "article",
            )
        )
    return papers


def test_filtering_agent_initialization(mock_request):
    """Test FilteringAgent initializes with cached embedding model"""
    agent = FilteringAgent(mock_request)

    assert agent.embedding_model is not None
    assert agent.embedding_model == mock_request.app.state.embedding_model


def test_filtering_agent_raises_error_without_model():
    """Test FilteringAgent raises error if embedding model not in app.state"""
    bad_request = Mock()
    bad_request.app.state = Mock(spec=[])  # No embedding_model attribute

    with pytest.raises(RuntimeError, match="Embedding model not found"):
        FilteringAgent(bad_request)


def test_two_pass_filtering_reduces_to_25(agent, sample_papers):
    """Test FilteringAgent reduces 100 papers to 25 through two-pass filtering"""
    state = State(
        original_query="CRISPR gene editing",
        optimized_query="CRISPR-Cas9 off-target effects",
        papers_metadata=sample_papers,
    )

    updated_state = agent.filter_candidates(state)

    assert len(updated_state.filtered_papers) == 25
    assert len(updated_state.papers_metadata) == 100  # Original unchanged


def test_filtering_skips_when_papers_under_25(agent):
    """Test FilteringAgent skips filtering if ≤25 papers"""
    papers = [
        Paper(
            title=f"Paper {i}",
            doi=f"10.1000/{i}",
            paper_id=f"{i}",
            authors=[],
            year=2023,
            journal="Test Journal",
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
        )
        for i in range(20)
    ]

    state = State(original_query="test", papers_metadata=papers)
    updated_state = agent.filter_candidates(state)

    # Should skip filtering and return all papers
    assert len(updated_state.filtered_papers) == 20
    assert updated_state.filtering_scores["skipped"] is True


def test_bm25_filtering_selects_relevant_papers(agent):
    """Test BM25 Pass 1 selects papers with high keyword overlap"""
    papers = [
        Paper(
            title="CRISPR gene editing in bacteria",
            abstract="CRISPR mechanisms",
            doi="1",
            paper_id="1",
            authors=[],
            year=2023,
            journal="Nature",
            pdf_url="https://example.com/1.pdf",
            type="article",
        ),
        Paper(
            title="Unrelated diabetes study",
            abstract="Insulin pathways",
            doi="2",
            paper_id="2",
            authors=[],
            year=2023,
            journal="Science",
            pdf_url="https://example.com/2.pdf",
            type="article",
        ),
        Paper(
            title="CRISPR Cas9 off-target effects",
            abstract="Gene editing accuracy",
            doi="3",
            paper_id="3",
            authors=[],
            year=2023,
            journal="Cell",
            pdf_url="https://example.com/3.pdf",
            type="article",
        ),
    ] * 20  # Repeat to get 60 papers

    query = "CRISPR gene editing"
    top_50, bm25_scores = agent._bm25_filter(papers, query, target_count=50)

    # Top papers should have higher BM25 scores
    assert len(top_50) == 50

    # Verify papers with "CRISPR" in title are ranked higher
    top_10_titles = [p.title for p in top_50[:10]]
    crispr_count = sum(1 for title in top_10_titles if "CRISPR" in title)
    assert crispr_count >= 5  # At least half should be CRISPR-related


def test_semantic_reranking_combines_scores(agent, sample_papers):
    """Test semantic Pass 2 combines semantic + citation + recency scores"""
    query = "CRISPR gene editing"

    # Use first 50 papers for reranking
    top_50 = sample_papers[:50]

    top_25, composite_scores = agent._semantic_rerank(top_50, query, target_count=25)

    assert len(top_25) == 25
    assert len(composite_scores) == 25

    # Scores should be in descending order
    assert composite_scores == sorted(composite_scores, reverse=True)

    # Scores should be between 0 and 1.2 (1.0 + 0.2 review boost)
    for score in composite_scores:
        assert 0 <= score <= 1.3


def test_review_papers_get_boost(agent):
    """Test review papers receive 20% boost in composite scoring"""
    papers = [
        Paper(
            title="Review",
            abstract="Abstract",
            doi="1",
            paper_id="1",
            authors=[],
            year=2023,
            journal="Nature Reviews",
            pdf_url="https://example.com/1.pdf",
            type="review",
            citation_count=100,
        ),
        Paper(
            title="Article",
            abstract="Abstract",
            doi="2",
            paper_id="2",
            authors=[],
            year=2023,
            journal="Nature",
            pdf_url="https://example.com/2.pdf",
            type="article",
            citation_count=100,
        ),
    ] * 25  # 50 papers total

    query = "test"
    top_25, scores = agent._semantic_rerank(papers, query, target_count=25)

    # Reviews should appear in top results due to boost
    review_count = sum(1 for p in top_25[:10] if p.type == "review")
    assert review_count >= 3  # At least 30% of top 10


def test_filtering_scores_structure(agent, sample_papers):
    """Test filtering_scores dict contains correct metrics"""
    state = State(original_query="test", papers_metadata=sample_papers)

    updated_state = agent.filter_candidates(state)
    scores = updated_state.filtering_scores

    assert "initial_count" in scores
    assert "bm25_filtered_count" in scores
    assert "final_count" in scores
    assert "bm25_time_ms" in scores
    assert "semantic_time_ms" in scores
    assert "total_time_ms" in scores
    assert "avg_bm25_score" in scores
    assert "avg_composite_score" in scores

    # Verify counts
    assert scores["initial_count"] == 100
    assert scores["bm25_filtered_count"] == 50
    assert scores["final_count"] == 25


def test_filtering_handles_missing_abstracts(agent):
    """Test FilteringAgent handles papers without abstracts"""
    papers = [
        Paper(
            title=f"Paper {i}",
            abstract=None,
            doi=f"10.1000/{i}",
            paper_id=f"{i}",
            authors=[],
            year=2023,
            journal="Test Journal",
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
            citation_count=10,
        )
        for i in range(100)
    ]

    state = State(original_query="test", papers_metadata=papers)

    # Should not crash, falls back to using title
    updated_state = agent.filter_candidates(state)

    assert len(updated_state.filtered_papers) == 25


@pytest.mark.performance
def test_filtering_performance(agent, sample_papers):
    """Test FilteringAgent completes in <250ms for 100 papers"""
    import time

    state = State(original_query="CRISPR gene editing", papers_metadata=sample_papers)

    start_time = time.time()
    agent.filter_candidates(state)
    elapsed = time.time() - start_time

    assert elapsed < 0.25, f"Filtering took {elapsed:.3f}s (target: <0.25s)"


@pytest.mark.performance
def test_filtering_scales_to_300_papers(agent):
    """Test FilteringAgent handles 300 papers within latency budget"""
    import time

    # Create 300 papers
    papers = [
        Paper(
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
            doi=f"10.1000/{i}",
            paper_id=f"{i}",
            authors=[],
            year=2023,
            journal="Test Journal",
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
            citation_count=10,
        )
        for i in range(300)
    ]

    state = State(original_query="test query", papers_metadata=papers)

    start_time = time.time()
    agent.filter_candidates(state)
    elapsed = time.time() - start_time

    # Should complete in <400ms even with 300 papers
    assert elapsed < 0.4, f"Filtering 300 papers took {elapsed:.3f}s (target: <0.4s)"


@pytest.mark.integration
def test_filtering_with_real_embedding_model():
    """Integration test with real SentenceTransformer model"""
    from sentence_transformers import SentenceTransformer

    # Create real request with actual model
    real_request = Mock()
    real_request.app.state.embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    agent = FilteringAgent(real_request)

    # Create test papers
    papers = [
        Paper(
            title=f"CRISPR gene editing study {i}",
            abstract=f"This paper investigates CRISPR-Cas9 mechanisms and applications {i}",
            doi=f"10.1000/{i}",
            paper_id=f"openalex:{i}",
            authors=["Author"],
            year=2023,
            journal="Nature",
            pdf_url=f"https://example.com/{i}.pdf",
            type="article",
            citation_count=100,
        )
        for i in range(100)
    ]

    state = State(
        original_query="CRISPR gene editing",
        optimized_query="CRISPR-Cas9 off-target effects and delivery mechanisms",
        papers_metadata=papers,
    )

    updated_state = agent.filter_candidates(state)

    # Verify results
    assert len(updated_state.filtered_papers) == 25
    assert updated_state.filtering_scores["total_time_ms"] < 500  # Should be fast
