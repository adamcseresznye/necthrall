import pytest

pytestmark = [pytest.mark.unit]
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
