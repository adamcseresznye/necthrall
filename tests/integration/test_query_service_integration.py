"""Integration tests for QueryService.

Tests the pipeline: query_optimization → semantic_scholar_search → quality_gate → composite_scoring.
Covers success paths, quality gate failures, and various error scenarios.
Tests the service layer directly without HTTP endpoints.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from config.config import get_settings
from models.state import Paper
from services.discovery_service import DiscoveryResult


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
def query_service(set_test_env):
    """Create QueryService with mocked embedding model."""
    # Mock the embedding model
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_dim = 384
    mock_embedding_model.encode.return_value = np.random.rand(384).astype(np.float32)

    # Mock get_text_embedding_batch to return list of embeddings
    def mock_get_embeddings(texts):
        return [np.random.rand(384).tolist() for _ in texts]

    mock_embedding_model.get_text_embedding_batch = mock_get_embeddings

    # Initialize query service with mock
    from services.query_service import QueryService

    return QueryService(get_settings(), mock_embedding_model)


@pytest.mark.integration
class TestQueryServiceIntegration:
    """Integration tests for QueryService.process_query() method."""

    @pytest.mark.asyncio
    async def test_successful_query_with_finalists(self, query_service):
        """Test case 1: Valid query returns finalists with all expected fields."""
        # Create mock papers
        mock_papers = [
            Paper(
                paperId=f"p{i}",
                title=f"Paper {i}",
                abstract="Test abstract",
                year=2023,
                citationCount=50,
                influentialCitationCount=10,
                openAccessPdf={"url": f"https://example.com/p{i}.pdf"},
                embedding={"specter": np.random.rand(384).tolist()},
                authors=[{"name": "Dr. Smith"}],
                venue="Test Journal",
            )
            for i in range(8)
        ]

        # Mock the discovery service to return success
        mock_discovery_result = DiscoveryResult(
            optimized_queries={
                "final_rephrase": "cardiovascular risks of intermittent fasting",
                "primary": "intermittent fasting cardiovascular risks",
                "broad": "fasting protocols health outcomes cardiovascular",
                "alternative": "time-restricted eating cardiac complications",
            },
            quality_gate={
                "passed": True,
                "metrics": {
                    "paper_count": 30,
                    "embedding_coverage": 1.0,
                    "abstract_coverage": 1.0,
                    "median_similarity": 0.8,
                },
                "reason": "Quality gate passed",
            },
            finalists=mock_papers,
            timing_breakdown={
                "query_optimization": 0.1,
                "semantic_scholar_search": 0.5,
                "quality_gate": 0.1,
                "composite_scoring": 0.2,
            },
            refinement_count=0,
        )

        with patch.object(
            query_service.discovery_service,
            "discover",
            return_value=mock_discovery_result,
        ):
            # Call service directly
            result = await query_service.process_query(
                "intermittent fasting cardiovascular risks", deep_mode=False
            )

            # Assert on PipelineResult object
            assert result.success is True
            assert result.query == "intermittent fasting cardiovascular risks"
            assert result.optimized_queries is not None
            assert result.quality_gate["passed"] is True
            assert len(result.finalists) == 8
            assert result.execution_time > 0
            assert "query_optimization" in result.timing_breakdown
            assert "semantic_scholar_search" in result.timing_breakdown

    @pytest.mark.asyncio
    async def test_quality_gate_failure_returns_early(self, query_service):
        """Test case 2: Quality gate failure returns early with reason (no finalists)."""
        # Mock the discovery service to return quality gate failure
        mock_discovery_result = DiscoveryResult(
            optimized_queries={
                "final_rephrase": "test query",
                "primary": "test query",
                "broad": "test query",
                "alternative": "test query",
            },
            quality_gate={
                "passed": False,
                "metrics": {
                    "paper_count": 0,
                    "embedding_coverage": 0.0,
                    "abstract_coverage": 0.0,
                    "median_similarity": 0.0,
                },
                "reason": "No papers found",
            },
            finalists=[],
            timing_breakdown={
                "query_optimization": 0.1,
                "semantic_scholar_search": 0.5,
                "quality_gate": 0.1,
            },
            refinement_count=1,
        )

        with patch.object(
            query_service.discovery_service,
            "discover",
            return_value=mock_discovery_result,
        ):
            # Call service directly
            result = await query_service.process_query("test query")

            # Assert on PipelineResult object
            assert result.success is True  # Pipeline handled the failure gracefully
            assert result.quality_gate["passed"] is False
            assert len(result.finalists) == 0
            assert "No papers found" in result.quality_gate["reason"]

    @pytest.mark.asyncio
    async def test_query_optimization_error_stops_pipeline(self, query_service):
        """Test case 3: Query optimization failure stops pipeline with error details."""
        # Import the custom exception
        from services.exceptions import QueryOptimizationError

        # Mock the discovery service to raise QueryOptimizationError
        with patch.object(
            query_service.discovery_service,
            "discover",
            side_effect=QueryOptimizationError(
                "Failed to optimize query: LLM service unavailable"
            ),
        ):
            # Call service directly
            result = await query_service.process_query("test query")

            # Assert on PipelineResult object
            assert result.success is False
            assert result.error_stage == "query_optimization"
            assert "Failed to optimize query" in result.error_message

    @pytest.mark.asyncio
    async def test_semantic_scholar_error_stops_pipeline(self, query_service):
        """Test case 4: Semantic Scholar API failure stops pipeline with error details."""
        # Import the custom exception
        from services.exceptions import SemanticScholarError

        # Mock the discovery service to raise SemanticScholarError
        with patch.object(
            query_service.discovery_service,
            "discover",
            side_effect=SemanticScholarError(
                "Semantic Scholar API is currently unavailable"
            ),
        ):
            # Call service directly
            result = await query_service.process_query("test query")

            # Assert on PipelineResult object
            assert result.success is False
            assert result.error_stage == "semantic_scholar_search"
            assert "Semantic Scholar API" in result.error_message

    @pytest.mark.asyncio
    async def test_quality_gate_error_stops_pipeline(self, query_service):
        """Test case 5: Quality gate validation error stops pipeline with error details."""
        # Import the custom exception
        from services.exceptions import QualityGateError

        # Mock the discovery service to raise QualityGateError
        with patch.object(
            query_service.discovery_service,
            "discover",
            side_effect=QualityGateError(
                "Failed to validate paper quality: Embedding validation failed"
            ),
        ):
            # Call service directly
            result = await query_service.process_query("test query")

            # Assert on PipelineResult object
            assert result.success is False
            assert result.error_stage == "quality_gate"
            assert "Failed to validate paper quality" in result.error_message

    @pytest.mark.asyncio
    async def test_ranking_error_stops_pipeline(self, query_service):
        """Test case 6: Ranking error stops pipeline with error details."""
        # Import the custom exception
        from services.exceptions import RankingError

        # Mock the discovery service to raise RankingError
        with patch.object(
            query_service.discovery_service,
            "discover",
            side_effect=RankingError("Failed to rank papers: Ranking algorithm failed"),
        ):
            # Call service directly
            result = await query_service.process_query("test query")

            # Assert on PipelineResult object
            assert result.success is False
            assert result.error_stage == "composite_scoring"
            assert "Failed to rank papers" in result.error_message

    @pytest.mark.asyncio
    async def test_unexpected_error_stops_pipeline(self, query_service):
        """Test case 7: Unexpected error stops pipeline with generic error message."""
        with patch.object(
            query_service.discovery_service,
            "discover",
            side_effect=Exception("Unexpected error"),
        ):
            # Call service directly
            result = await query_service.process_query("test query")

            # Assert on PipelineResult object
            assert result.success is False
            assert result.error_message is not None
            assert "Unexpected error" in result.error_message
