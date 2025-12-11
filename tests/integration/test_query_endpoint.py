"""Integration tests for the /query endpoint.

Tests the pipeline: query_optimization → semantic_scholar_search → quality_gate → composite_scoring.
Covers success paths, quality gate failures, and various error scenarios.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import os


@pytest.fixture
def set_test_env():
    """Set minimal test environment variables for config validation"""
    # Save original values
    original_values = {
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        "PRIMARY_LLM_API_KEY": os.environ.get("PRIMARY_LLM_API_KEY"),
        "SECONDARY_LLM_API_KEY": os.environ.get("SECONDARY_LLM_API_KEY"),
        "QUERY_OPTIMIZATION_MODEL": os.environ.get("QUERY_OPTIMIZATION_MODEL"),
        "SYNTHESIS_MODEL": os.environ.get("SYNTHESIS_MODEL"),
        "RAG_EMBEDDING_MODEL": os.environ.get("RAG_EMBEDDING_MODEL"),
        "SKIP_DOTENV_LOADER": os.environ.get("SKIP_DOTENV_LOADER"),
    }

    # Set test values
    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "test_key"
    os.environ["PRIMARY_LLM_API_KEY"] = "test_key"
    os.environ["SECONDARY_LLM_API_KEY"] = "test_key"
    os.environ["QUERY_OPTIMIZATION_MODEL"] = "test_model"
    os.environ["SYNTHESIS_MODEL"] = "test_model"
    os.environ["RAG_EMBEDDING_MODEL"] = "allenai-specter-2"
    os.environ["SKIP_DOTENV_LOADER"] = "1"

    yield

    # Restore original values
    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def app(set_test_env):
    from main import app

    # Mock the embedding model
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = np.random.rand(384).astype(np.float32)
    app.state.embedding_model = mock_embedding_model

    # Initialize query service with mock
    from services.query_service import QueryService

    app.state.query_service = QueryService(mock_embedding_model)

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.integration
class TestQueryEndpoint:
    """Integration tests for the /query POST endpoint."""

    def test_successful_query_with_finalists(self, client: TestClient):
        """Test case 1: Valid query returns finalists with all expected fields."""
        # Mock all pipeline components for success path
        with (
            patch("services.query_service.QueryOptimizationAgent") as mock_opt_class,
            patch("services.query_service.SemanticScholarClient") as mock_client_class,
            patch("services.query_service.validate_quality") as mock_quality,
            patch("services.query_service.RankingAgent") as mock_ranker_class,
        ):

            # Setup mocks
            mock_opt = MagicMock()
            mock_opt.generate_dual_queries = AsyncMock(
                return_value={
                    "final_rephrase": "cardiovascular risks of intermittent fasting",
                    "primary": "intermittent fasting cardiovascular risks",
                    "broad": "fasting protocols health outcomes cardiovascular",
                    "alternative": "time-restricted eating cardiac complications",
                }
            )
            mock_opt_class.return_value = mock_opt

            mock_client = MagicMock()
            mock_papers = [
                {
                    "paperId": "p1",
                    "title": "Effects of Intermittent Fasting on Cardiovascular Health",
                    "abstract": "This study examines...",
                    "year": 2023,
                    "citationCount": 50,
                    "influentialCitationCount": 10,
                    "openAccessPdf": {"url": "https://example.com/p1.pdf"},
                    "embedding": {"specter": np.random.rand(384).tolist()},
                    "authors": [{"name": "Dr. Smith"}],
                    "venue": "Journal of Cardiology",
                }
            ] * 30  # 30 papers to pass quality gate
            mock_client.multi_query_search = AsyncMock(return_value=mock_papers)
            mock_client_class.return_value = mock_client

            mock_quality.return_value = {
                "passed": True,
                "metrics": {
                    "paper_count": 30,
                    "embedding_coverage": 1.0,
                    "abstract_coverage": 1.0,
                    "median_similarity": 0.8,
                },
                "reason": "Quality gate passed",
            }

            mock_ranker = MagicMock()
            mock_ranker.rank_papers.return_value = mock_papers[:8]  # Top 8 finalists
            mock_ranker_class.return_value = mock_ranker

            # Make request
            response = client.post(
                "/query", json={"query": "intermittent fasting cardiovascular risks"}
            )

            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert data["query"] == "intermittent fasting cardiovascular risks"
            assert "optimized_queries" in data
            assert "quality_gate" in data
            assert data["quality_gate"]["passed"] is True
            assert len(data["finalists"]) == 8
            assert data["execution_time"] > 0
            assert "timing_breakdown" in data
            assert all(
                k in data["timing_breakdown"]
                for k in [
                    "query_optimization",
                    "semantic_scholar_search",
                    "quality_gate",
                    "composite_scoring",
                ]
            )

    def test_quality_gate_failure_returns_early(self, client: TestClient):
        """Test case 2: Quality gate failure returns early with reason (no finalists)."""
        with (
            patch("services.query_service.QueryOptimizationAgent") as mock_opt_class,
            patch("services.query_service.SemanticScholarClient") as mock_client_class,
            patch("services.query_service.validate_quality") as mock_quality,
        ):

            mock_opt = MagicMock()
            mock_opt.generate_dual_queries = AsyncMock(
                return_value={
                    "final_rephrase": "test query",
                    "primary": "test query",
                    "broad": "test query",
                    "alternative": "test query",
                }
            )
            mock_opt_class.return_value = mock_opt

            mock_client = MagicMock()
            mock_client.multi_query_search = AsyncMock(return_value=[])  # No papers
            mock_client_class.return_value = mock_client

            mock_quality.return_value = {
                "passed": False,
                "metrics": {
                    "paper_count": 0,
                    "embedding_coverage": 0.0,
                    "abstract_coverage": 0.0,
                    "median_similarity": 0.0,
                },
                "reason": "insufficient paper count (0 < 25)",
            }

            response = client.post("/query", json={"query": "test query"})

            assert response.status_code == 200
            data = response.json()

            assert data["quality_gate"]["passed"] is False
            assert data["finalists"] == []
            assert "insufficient paper count" in data["quality_gate"]["reason"]

    def test_query_optimization_error_returns_500(self, client: TestClient):
        """Test case 3: Query optimization failure returns 500 with user-friendly message."""
        with patch("services.query_service.QueryOptimizationAgent") as mock_opt_class:
            mock_opt = MagicMock()
            mock_opt.generate_dual_queries = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )
            mock_opt_class.return_value = mock_opt

            response = client.post("/query", json={"query": "test query"})

            assert response.status_code == 500
            data = response.json()
            assert "Failed to optimize query" in data["detail"]

    def test_semantic_scholar_error_returns_503(self, client: TestClient):
        """Test case 4: Semantic Scholar API failure returns 503 with user-friendly message."""
        with (
            patch("services.query_service.QueryOptimizationAgent") as mock_opt_class,
            patch("services.query_service.SemanticScholarClient") as mock_client_class,
        ):

            mock_opt = MagicMock()
            mock_opt.generate_dual_queries = AsyncMock(
                return_value={
                    "final_rephrase": "test query",
                    "primary": "test query",
                    "broad": "test query",
                    "alternative": "test query",
                }
            )
            mock_opt_class.return_value = mock_opt

            mock_client = MagicMock()
            mock_client.multi_query_search = AsyncMock(
                side_effect=Exception("Semantic Scholar API unavailable")
            )
            mock_client_class.return_value = mock_client

            response = client.post("/query", json={"query": "test query"})

            assert response.status_code == 503
            data = response.json()
            assert "Semantic Scholar API is currently unavailable" in data["detail"]

    def test_quality_gate_error_returns_500(self, client: TestClient):
        """Test case 5: Quality gate validation error returns 500."""
        with (
            patch("services.query_service.QueryOptimizationAgent") as mock_opt_class,
            patch("services.query_service.SemanticScholarClient") as mock_client_class,
            patch("services.query_service.validate_quality") as mock_quality,
        ):

            mock_opt = MagicMock()
            mock_opt.generate_dual_queries = AsyncMock(
                return_value={
                    "final_rephrase": "test query",
                    "primary": "test query",
                    "broad": "test query",
                    "alternative": "test query",
                }
            )
            mock_opt_class.return_value = mock_opt

            mock_client = MagicMock()
            mock_client.multi_query_search = AsyncMock(
                return_value=[{"paperId": "test"}] * 30
            )
            mock_client_class.return_value = mock_client

            mock_quality.side_effect = Exception("Embedding validation failed")

            response = client.post("/query", json={"query": "test query"})

            assert response.status_code == 500
            data = response.json()
            assert "Failed to validate paper quality" in data["detail"]

    def test_ranking_error_returns_500(self, client: TestClient):
        """Test case 6: Ranking error returns 500."""
        with (
            patch("services.query_service.QueryOptimizationAgent") as mock_opt_class,
            patch("services.query_service.SemanticScholarClient") as mock_client_class,
            patch("services.query_service.validate_quality") as mock_quality,
            patch("services.query_service.RankingAgent") as mock_ranker_class,
        ):

            mock_opt = MagicMock()
            mock_opt.generate_dual_queries = AsyncMock(
                return_value={
                    "final_rephrase": "test query",
                    "primary": "test query",
                    "broad": "test query",
                    "alternative": "test query",
                }
            )
            mock_opt_class.return_value = mock_opt

            mock_client = MagicMock()
            mock_client.multi_query_search = AsyncMock(
                return_value=[{"paperId": "test"}] * 30
            )
            mock_client_class.return_value = mock_client

            mock_quality.return_value = {
                "passed": True,
                "metrics": {
                    "paper_count": 30,
                    "embedding_coverage": 1.0,
                    "abstract_coverage": 1.0,
                    "median_similarity": 0.8,
                },
                "reason": "Quality gate passed",
            }

            mock_ranker = MagicMock()
            mock_ranker.rank_papers.side_effect = Exception("Ranking algorithm failed")
            mock_ranker_class.return_value = mock_ranker

            response = client.post("/query", json={"query": "test query"})

            assert response.status_code == 500
            data = response.json()
            assert "Failed to rank papers" in data["detail"]

    def test_empty_query_returns_422(self, client: TestClient):
        """Test case 7: Empty query returns 422 Bad Request."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_query_too_long_returns_422(self, client: TestClient):
        """Test case 8: Query exceeding 500 characters returns 422 Bad Request."""
        long_query = "a" * 501
        response = client.post("/query", json={"query": long_query})
        assert response.status_code == 422  # Pydantic validation error

    def test_unexpected_error_returns_500(self, client: TestClient):
        """Test case 9: Unexpected error returns 500 with generic message."""
        with patch.object(
            client.app.state.query_service,
            "process_query",
            side_effect=Exception("Unexpected error"),
        ):
            response = client.post("/query", json={"query": "test query"})

            assert response.status_code == 500
            data = response.json()
            assert "unexpected error occurred" in data["detail"].lower()
