import pytest
import time
import requests
import json
from unittest.mock import patch, Mock
from utils.openalex_client import OpenAlexClient, Paper, _reconstruct_abstract


@pytest.fixture
def client():
    """Fixture for OpenAlexClient."""
    return OpenAlexClient()


def test_successful_search(client):
    """Test a successful search returns Paper objects."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "authorships": [{"author": {"display_name": "Dr. Test"}}],
                    "publication_year": 2023,
                    "primary_location": {"source": {"display_name": "Test Journal"}},
                    "cited_by_count": 10,
                    "doi": "https://doi.org/10.1000/test",
                    "best_oa_location": {"pdf_url": "http://example.com/test.pdf"},
                    "abstract_inverted_index": {
                        "This": [0],
                        "is": [1],
                        "a": [2],
                        "test": [3],
                    },
                    "type": "article",
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = client.search_papers("test query", max_results=1)
        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, Paper)
        assert paper.paper_id == "https://openalex.org/W123"
        assert paper.title == "Test Paper"
        assert paper.authors == ["Dr. Test"]
        assert paper.year == 2023
        assert paper.journal == "Test Journal"
        assert paper.citation_count == 10
        assert paper.pdf_url == "http://example.com/test.pdf"
        assert paper.abstract == "This is a test"
        assert paper.type == "article"


def test_rate_limiting(client):
    """Test that rate limiting enforces a delay."""
    with patch("time.sleep") as mock_sleep:
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"results": []}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            client.search_papers("test")
            mock_sleep.assert_any_call(0.1)


def test_api_timeout(client):
    """Test that an API timeout returns an empty list."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout):
        papers = client.search_papers("test")
        assert papers == []


def test_http_429_retry(client):
    """Test that a 429 error triggers a retry."""
    with patch("requests.get") as mock_get:
        mock_429_response = Mock()
        mock_429_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=Mock(status_code=429)
        )

        mock_success_response = Mock()
        mock_success_response.json.return_value = {"results": []}
        mock_success_response.raise_for_status = Mock()

        mock_get.side_effect = [mock_429_response, mock_success_response]

        with patch("time.sleep") as mock_sleep:
            client.search_papers("test", retries=2, backoff_factor=0.1)
            mock_sleep.assert_any_call(0.1 * (2**0))


def test_server_error_retry(client):
    """Test that a 500 error triggers a retry."""
    with patch("requests.get") as mock_get:
        mock_500_response = Mock()
        mock_500_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=Mock(status_code=500)
        )

        mock_success_response = Mock()
        mock_success_response.json.return_value = {"results": []}
        mock_success_response.raise_for_status = Mock()

        mock_get.side_effect = [mock_500_response, mock_success_response]

        with patch("time.sleep") as mock_sleep:
            client.search_papers("test", retries=2)
            mock_sleep.assert_any_call(1)


def test_malformed_json_response(client):
    """Test that a malformed JSON response returns an empty list."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = client.search_papers("test")
        assert papers == []


def test_empty_results(client):
    """Test that an empty result set from the API returns an empty list."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = client.search_papers("test")
        assert papers == []


def test_paper_with_no_authors(client):
    """Test that a paper with no authors is parsed correctly."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "authorships": [],
                    "publication_year": 2023,
                    "primary_location": {"source": {"display_name": "Test Journal"}},
                    "cited_by_count": 10,
                    "doi": "https://doi.org/10.1000/test",
                    "best_oa_location": {"pdf_url": "http://example.com/test.pdf"},
                    "abstract_inverted_index": None,
                    "type": "article",
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = client.search_papers("test query", max_results=1)
        assert len(papers) == 1
        assert papers[0].authors == []


def test_paper_with_no_journal(client):
    """Test that a paper with no journal is parsed correctly."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "authorships": [],
                    "publication_year": 2023,
                    "primary_location": None,
                    "cited_by_count": 10,
                    "doi": "https://doi.org/10.1000/test",
                    "best_oa_location": {"pdf_url": "http://example.com/test.pdf"},
                    "abstract_inverted_index": None,
                    "type": "article",
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = client.search_papers("test query", max_results=1)
        assert len(papers) == 1
        assert papers[0].journal == "Unknown"


def test_paper_with_no_pdf_url(client):
    """Test that a paper with no pdf_url is skipped."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "authorships": [],
                    "publication_year": 2023,
                    "primary_location": None,
                    "cited_by_count": 10,
                    "best_oa_location": {"pdf_url": None},
                    "abstract_inverted_index": None,
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        papers = client.search_papers("test query", max_results=1)
        assert len(papers) == 0


def test_reconstruct_abstract():
    """Test abstract reconstruction."""
    inverted_index = {
        "This": [0],
        "is": [1],
        "a": [2],
        "test": [3],
        "of": [4],
        "the": [5],
        "abstract": [6],
        "reconstruction": [7],
    }
    expected_abstract = "This is a test of the abstract reconstruction"
    assert _reconstruct_abstract(inverted_index) == expected_abstract


def test_reconstruct_abstract_empty():
    """Test abstract reconstruction with empty input."""
    assert _reconstruct_abstract(None) is None
    assert _reconstruct_abstract({}) == ""
