import pytest

pytestmark = [pytest.mark.integration]
import inspect
from unittest.mock import patch, MagicMock, Mock
from agents.search import SearchAgent
from models.state import State, Paper


@pytest.fixture
def search_agent():
    with patch("agents.search.OpenAlexClient") as mock_openalex:
        agent = SearchAgent()
        agent.openalex_client = MagicMock()
        return agent


@pytest.fixture
def mock_state():
    return State(
        original_query="CRISPR gene editing",
        optimized_query="CRISPR-Cas9 off-target effects in mammalian cells",
    )


def test_search_agent_uses_optimized_query(search_agent, mock_state):
    """Test SearchAgent uses optimized_query instead of original_query"""
    with patch.object(search_agent, "_query_openalex", return_value=[]):
        search_agent.search(mock_state)

        # Verify _query_openalex was called with optimized query
        calls = search_agent._query_openalex.call_args_list
        assert any("CRISPR-Cas9 off-target effects" in str(call) for call in calls)


def test_search_agent_queries_two_paper_types(search_agent, mock_state):
    """Test SearchAgent queries both 'review' and 'article' types"""
    with patch.object(search_agent, "_query_openalex", return_value=[]) as mock_query:
        search_agent.search(mock_state)

        # Should have 2 calls: one for reviews, one for articles
        assert mock_query.call_count == 2

        # Check the arguments of each call
        call_args_list = mock_query.call_args_list
        paper_types = []
        for call in call_args_list:
            args, kwargs = call
            paper_types.append(
                kwargs["paper_type"]
            )  # paper_type is passed as keyword argument

        assert "review" in paper_types
        assert "article" in paper_types


def test_search_agent_retrieves_correct_paper_counts(search_agent, mock_state):
    """Test SearchAgent retrieves 30 reviews + 70 articles = 100 total"""
    from agents.search import DataclassPaper

    mock_reviews = [
        DataclassPaper(
            paper_id=f"openalex:review:{i}",
            title=f"Review paper {i}",
            authors=[f"Author {i}"],
            year=2023,
            journal="Review Journal",
            citation_count=50,
            doi=f"10.1000/review/{i}",
            abstract=f"Review abstract {i}",
            pdf_url=f"https://example.com/review/{i}.pdf",
            type="review",
        )
        for i in range(30)
    ]
    mock_articles = [
        DataclassPaper(
            paper_id=f"openalex:article:{i}",
            title=f"Article paper {i}",
            authors=[f"Author {i}"],
            year=2023,
            journal="Article Journal",
            citation_count=25,
            doi=f"10.1000/article/{i}",
            abstract=f"Article abstract {i}",
            pdf_url=f"https://example.com/article/{i}.pdf",
            type="article",
        )
        for i in range(70)
    ]

    def mock_query(query, paper_type, limit):
        return mock_reviews if paper_type == "review" else mock_articles

    with patch.object(search_agent, "_query_openalex", side_effect=mock_query):
        updated_state = search_agent.search(mock_state)

        assert len(updated_state.papers_metadata) == 100
        review_count = sum(
            1 for p in updated_state.papers_metadata if p.type == "review"
        )
        article_count = sum(
            1 for p in updated_state.papers_metadata if p.type == "article"
        )

        assert review_count == 30
        assert article_count == 70


def test_search_quality_assessment_pass(search_agent, mock_state):
    """Test search quality passes with sufficient papers and relevance"""
    # Mock 50 papers with high relevance
    mock_papers = [
        Paper(
            paper_id=f"openalex:{i}",
            title=f"CRISPR-Cas9 gene editing paper {i}",
            authors=["Author"],
            year=2023,
            journal="Nature",
            citation_count=100,
            doi=f"10.1000/{i}",
            abstract="Abstract",
            pdf_url="https://example.com/paper.pdf",
            type="article",
        )
        for i in range(50)
    ]

    quality = search_agent._assess_quality(mock_papers, "CRISPR-Cas9 gene editing")

    assert quality["passed"] is True
    assert quality["paper_count"] == 50
    assert quality["avg_relevance"] >= 0.4


def test_search_quality_assessment_fail_insufficient_papers(search_agent):
    """Test search quality fails with <10 papers"""
    from agents.search import DataclassPaper

    mock_papers = [
        DataclassPaper(
            paper_id=f"openalex:{i}",
            title="Unrelated paper",
            authors=["Author"],
            year=2023,
            journal="Journal",
            citation_count=10,
            doi=f"10.1000/{i}",
            abstract="Abstract",
            pdf_url="https://example.com/paper.pdf",
            type="article",
        )
        for i in range(5)
    ]

    quality = search_agent._assess_quality(mock_papers, "CRISPR")

    assert quality["passed"] is False
    assert quality["paper_count"] == 5
    assert "Insufficient results" in quality["reason"]


def test_search_quality_assessment_fail_low_relevance(search_agent):
    """Test search quality fails with low avg_relevance"""
    # Papers with titles unrelated to query
    mock_papers = [
        Paper(
            paper_id=f"openalex:{i}",
            title=f"Unrelated topic paper {i}",
            authors=["Author"],
            year=2023,
            journal="Journal",
            citation_count=10,
            doi=f"10.1000/{i}",
            abstract="Abstract",
            pdf_url="https://example.com/paper.pdf",
            type="article",
        )
        for i in range(20)
    ]

    quality = search_agent._assess_quality(mock_papers, "CRISPR gene editing")

    assert quality["passed"] is False
    assert quality["avg_relevance"] < 0.4


def test_no_arxiv_code_present(search_agent):
    """Test arXiv integration has been completely removed"""
    agent_source = inspect.getsource(SearchAgent)

    assert "arxiv" not in agent_source.lower()
    assert "_query_arxiv" not in agent_source
    assert "arxiv.org" not in agent_source


@patch("utils.openalex_client.requests.get")
def test_openalex_api_request_format(mock_get, search_agent):
    """Test OpenAlex API is called with correct parameters"""
    mock_response = Mock()
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    search_agent._query_openalex("CRISPR", "article", 70)

    # Verify API call
    call_args = mock_get.call_args
    if call_args:
        assert "api.openalex.org/works" in call_args[0][0]

        params = call_args[1]["params"]
        assert params["search"] == "CRISPR"
        assert "type:article" in params["filter"]
        assert "is_oa:true" in params["filter"]
        assert params["per_page"] == 70


def test_search_agent_end_to_end():
    """Test SearchAgent completes full search with real OpenAlex API"""
    agent = SearchAgent()
    state = State(
        original_query="intermittent fasting",
        optimized_query="metabolic effects of intermittent fasting protocols",
    )

    updated_state = agent.search(state)

    # Verify search quality assessment is present
    assert updated_state.search_quality is not None
    assert "passed" in updated_state.search_quality
    assert "reason" in updated_state.search_quality
    assert "paper_count" in updated_state.search_quality
    assert "avg_relevance" in updated_state.search_quality

    # If papers are found, verify they have required fields
    if len(updated_state.papers_metadata) > 0:
        # Verify all papers have required fields
        for paper in updated_state.papers_metadata:
            assert paper.title
            assert paper.doi or paper.paper_id
            assert paper.pdf_url  # All should have PDF access

        # Verify paper types
        review_count = sum(
            1 for p in updated_state.papers_metadata if p.type == "review"
        )
        article_count = sum(
            1 for p in updated_state.papers_metadata if p.type == "article"
        )

        assert review_count > 0 or article_count > 0, "Should have at least some papers"
    else:
        # If no papers found, search quality should indicate failure
        assert updated_state.search_quality["passed"] is False
        assert "Insufficient results" in updated_state.search_quality["reason"]
