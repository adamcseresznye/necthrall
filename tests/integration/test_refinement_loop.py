"""Integration tests for the QueryService refinement loop.

Tests the automatic retry/refinement behavior when QualityGate fails
on the first attempt.

Test Coverage:
    - Verify refinement triggers when first quality gate fails
    - Verify refinement_count becomes 1 on successful retry
    - Verify pipeline returns failure state if both attempts fail
    - Verify no infinite loops (max 1 retry)

Run these tests with:
    pytest tests/integration/test_refinement_loop.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from services.query_service import QueryService, PipelineResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_optimizer():
    """Create a mock QueryOptimizationAgent."""
    optimizer = AsyncMock()
    optimizer.generate_dual_queries = AsyncMock(
        return_value={
            "primary": "cardiovascular effects intermittent fasting",
            "broad": "fasting heart health benefits",
            "alternative": "time-restricted eating cardiovascular",
            "final_rephrase": "cardiovascular effects of intermittent fasting",
        }
    )
    return optimizer


@pytest.fixture
def low_quality_papers():
    """Papers that will fail quality gate (low citations, old, no venue)."""
    return [
        {
            "paperId": "bad_paper_1",
            "title": "Irrelevant Study",
            "citationCount": 0,
            "year": 1990,
            "venue": None,
            "authors": [],
            "abstract": "Not relevant",
        },
        {
            "paperId": "bad_paper_2",
            "title": "Another Irrelevant Study",
            "citationCount": 1,
            "year": 1985,
            "venue": "",
            "authors": [],
            "abstract": "Also not relevant",
        },
    ]


@pytest.fixture
def high_quality_papers():
    """Papers that will pass quality gate (good citations, recent, good venue)."""
    import numpy as np

    # Create mock embedding (384-dim to match expected SPECTER2 dimensions)
    mock_embedding = np.random.rand(384).tolist()

    return [
        {
            "paperId": "good_paper_1",
            "title": "Cardiovascular Effects of Intermittent Fasting",
            "citationCount": 150,
            "year": 2023,
            "venue": "Nature Medicine",
            "authors": [{"name": "Dr. Smith"}],
            "abstract": "We studied the cardiovascular effects of intermittent fasting...",
            "embedding": {"vector": mock_embedding},
        },
        {
            "paperId": "good_paper_2",
            "title": "Heart Health and Fasting Protocols",
            "citationCount": 200,
            "year": 2022,
            "venue": "JAMA Cardiology",
            "authors": [{"name": "Dr. Jones"}],
            "abstract": "This meta-analysis examines fasting and heart health...",
            "embedding": {"vector": mock_embedding},
        },
        {
            "paperId": "good_paper_3",
            "title": "Time-Restricted Eating Benefits",
            "citationCount": 100,
            "year": 2023,
            "venue": "Cell Metabolism",
            "authors": [{"name": "Dr. Brown"}],
            "abstract": "Time-restricted eating shows significant benefits...",
            "embedding": {"vector": mock_embedding},
        },
    ]


@pytest.fixture
def mock_client(low_quality_papers):
    """Create a mock SemanticScholarClient."""
    client = AsyncMock()
    client.multi_query_search = AsyncMock(return_value=low_quality_papers)
    return client


@pytest.fixture
def mock_ranker(high_quality_papers):
    """Create a mock RankingAgent."""
    ranker = MagicMock()
    ranker.rank_papers.return_value = high_quality_papers
    return ranker


@pytest.fixture
def query_service(mock_optimizer, mock_client, mock_ranker):
    """Create a QueryService instance with mocked agents."""
    service = QueryService()

    # Inject mocks directly into discovery service
    service.discovery_service._optimizer = mock_optimizer
    service.discovery_service._client = mock_client
    service.discovery_service._ranker = mock_ranker

    # Mock other components we don't need for this test
    service.ingestion_service._acquisition_agent = MagicMock()
    service.ingestion_service._processing_agent = MagicMock()
    service.rag_service._retriever = MagicMock()
    service.rag_service._reranker = MagicMock()
    service.rag_service._synthesis_agent = MagicMock()
    service.rag_service._verifier = MagicMock()

    return service


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_refinement_triggers_on_quality_gate_failure(
    mock_optimizer, low_quality_papers, high_quality_papers
):
    """Test that refinement triggers when first quality gate fails.

    Scenario:
        1. First search returns low-quality papers → quality gate fails
        2. Refinement triggers with broad query
        3. Second search returns high-quality papers → quality gate passes
        4. refinement_count should be 1
    """
    # Create service with no embedding model
    service = QueryService(embedding_model=None)

    # Mock the optimizer
    service.discovery_service._optimizer = mock_optimizer

    # Mock the Semantic Scholar client to return bad papers first, good papers second
    mock_client = AsyncMock()
    call_count = 0

    async def mock_search(queries, limit_per_query=100):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt: return low-quality papers
            return low_quality_papers
        else:
            # Refinement attempt: return high-quality papers
            return high_quality_papers

    mock_client.multi_query_search = mock_search
    service.discovery_service._client = mock_client

    # Mock quality gate to fail on first call (low_quality_papers), pass on second (high_quality_papers)
    quality_call_count = 0

    def mock_validate_quality(papers, query_embedding):
        nonlocal quality_call_count
        quality_call_count += 1
        if quality_call_count == 1:
            # First attempt: fail quality gate
            return {
                "passed": False,
                "metrics": {
                    "paper_count": 2,
                    "embedding_coverage": 0.0,
                    "abstract_coverage": 1.0,
                },
                "reason": "insufficient paper count (2 < 25)",
            }
        else:
            # Refinement attempt: pass quality gate
            return {
                "passed": True,
                "metrics": {
                    "paper_count": 30,
                    "embedding_coverage": 0.8,
                    "abstract_coverage": 1.0,
                },
                "reason": "",
            }

    with patch(
        "services.discovery_service.validate_quality", side_effect=mock_validate_quality
    ):
        # Run the pipeline
        result = await service.process_query(
            "What are the cardiovascular effects of intermittent fasting?"
        )

    # Assertions
    assert result.success is True, "Pipeline should succeed after refinement"
    assert result.refinement_count == 1, "refinement_count should be 1 after one retry"
    assert (
        len(result.finalists) > 0
    ), "Should have finalists after successful refinement"
    assert call_count == 2, "Should have made exactly 2 search calls"

    # Verify timing breakdown includes refinement stages
    assert "semantic_scholar_search" in result.timing_breakdown
    assert "semantic_scholar_search_refinement" in result.timing_breakdown
    assert "quality_gate" in result.timing_breakdown
    assert "quality_gate_refinement" in result.timing_breakdown


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_refinement_when_first_attempt_succeeds(
    mock_optimizer, high_quality_papers
):
    """Test that no refinement occurs when first quality gate passes.

    Scenario:
        1. First search returns high-quality papers → quality gate passes
        2. No refinement should trigger
        3. refinement_count should be 0
    """
    service = QueryService(embedding_model=None)
    service.discovery_service._optimizer = mock_optimizer

    mock_client = AsyncMock()
    mock_client.multi_query_search = AsyncMock(return_value=high_quality_papers)
    service.discovery_service._client = mock_client

    # Mock quality gate to pass on first call
    def mock_validate_quality(papers, query_embedding):
        return {
            "passed": True,
            "metrics": {
                "paper_count": 30,
                "embedding_coverage": 0.8,
                "abstract_coverage": 1.0,
            },
            "reason": "",
        }

    with patch(
        "services.discovery_service.validate_quality", side_effect=mock_validate_quality
    ):
        result = await service.process_query(
            "What are the cardiovascular effects of intermittent fasting?"
        )

    assert result.success is True
    assert (
        result.refinement_count == 0
    ), "refinement_count should be 0 when no retry needed"
    assert len(result.finalists) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_returns_failure_after_both_attempts_fail(
    mock_optimizer, low_quality_papers
):
    """Test that pipeline returns correct state when both attempts fail.

    Scenario:
        1. First search returns low-quality papers → quality gate fails
        2. Refinement triggers with broad query
        3. Second search also returns low-quality papers → quality gate fails again
        4. Pipeline should still return success=True but with empty finalists
        5. refinement_count should be 1
    """
    service = QueryService(embedding_model=None)
    service.discovery_service._optimizer = mock_optimizer

    # Mock client to always return low-quality papers
    mock_client = AsyncMock()
    mock_client.multi_query_search = AsyncMock(return_value=low_quality_papers)
    service.discovery_service._client = mock_client

    # Mock quality gate to always fail
    def mock_validate_quality(papers, query_embedding):
        return {
            "passed": False,
            "metrics": {
                "paper_count": 2,
                "embedding_coverage": 0.0,
                "abstract_coverage": 1.0,
            },
            "reason": "insufficient paper count (2 < 25)",
        }

    with patch(
        "services.discovery_service.validate_quality", side_effect=mock_validate_quality
    ):
        result = await service.process_query(
            "What are the cardiovascular effects of intermittent fasting?"
        )

    # Pipeline still "succeeds" but with no finalists
    assert result.success is True
    assert result.refinement_count == 1, "Should have attempted exactly 1 refinement"
    assert (
        len(result.finalists) == 0
    ), "Should have no finalists when quality gate fails"
    assert result.quality_gate["passed"] is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_max_one_refinement_attempt(mock_optimizer, low_quality_papers):
    """Test that only one refinement attempt is made (no infinite loops).

    Scenario:
        1. First search fails quality gate
        2. One refinement attempt is made
        3. No further refinements even if second attempt fails
    """
    service = QueryService(embedding_model=None)
    service.discovery_service._optimizer = mock_optimizer

    call_count = 0

    async def mock_search(queries, limit_per_query=100):
        nonlocal call_count
        call_count += 1
        return low_quality_papers

    mock_client = AsyncMock()
    mock_client.multi_query_search = mock_search
    service.discovery_service._client = mock_client

    # Mock quality gate to always fail
    def mock_validate_quality(papers, query_embedding):
        return {
            "passed": False,
            "metrics": {
                "paper_count": 2,
                "embedding_coverage": 0.0,
                "abstract_coverage": 1.0,
            },
            "reason": "insufficient paper count",
        }

    with patch(
        "services.discovery_service.validate_quality", side_effect=mock_validate_quality
    ):
        result = await service.process_query(
            "What are the cardiovascular effects of intermittent fasting?"
        )

    # Should have made exactly 2 calls (initial + 1 refinement)
    assert (
        call_count == 2
    ), "Should make exactly 2 search calls (initial + 1 refinement)"
    assert result.refinement_count == 1, "refinement_count should be 1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_refinement_uses_broad_query(
    mock_optimizer, low_quality_papers, high_quality_papers
):
    """Test that refinement uses the broad query as fallback.

    Verify the queries passed to the second search include the broad query.
    """
    service = QueryService(embedding_model=None)
    service.discovery_service._optimizer = mock_optimizer

    captured_queries = []

    async def mock_search(queries, limit_per_query=100):
        captured_queries.append(queries)
        if len(captured_queries) == 1:
            return low_quality_papers
        return high_quality_papers

    mock_client = AsyncMock()
    mock_client.multi_query_search = mock_search
    service.discovery_service._client = mock_client

    # Mock quality gate: fail first, pass second
    quality_call_count = 0

    def mock_validate_quality(papers, query_embedding):
        nonlocal quality_call_count
        quality_call_count += 1
        if quality_call_count == 1:
            return {
                "passed": False,
                "metrics": {
                    "paper_count": 2,
                    "embedding_coverage": 0.0,
                    "abstract_coverage": 1.0,
                },
                "reason": "insufficient paper count",
            }
        return {
            "passed": True,
            "metrics": {
                "paper_count": 30,
                "embedding_coverage": 0.8,
                "abstract_coverage": 1.0,
            },
            "reason": "",
        }

    with patch(
        "services.discovery_service.validate_quality", side_effect=mock_validate_quality
    ):
        result = await service.process_query(
            "What are the cardiovascular effects of intermittent fasting?"
        )

    assert len(captured_queries) == 2, "Should have captured 2 sets of queries"

    # Second call should include the broad query
    refinement_queries = captured_queries[1]
    assert (
        "fasting heart health benefits" in refinement_queries
    ), "Refinement should use the broad query"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_refinement_count_in_pipeline_result_dataclass():
    """Test that PipelineResult has refinement_count field with default value."""
    result = PipelineResult(
        query="test",
        optimized_queries={},
        quality_gate={},
        finalists=[],
        execution_time=1.0,
        timing_breakdown={},
        success=True,
    )

    # Default value should be 0
    assert result.refinement_count == 0

    # Should be settable
    result_with_refinement = PipelineResult(
        query="test",
        optimized_queries={},
        quality_gate={},
        finalists=[],
        execution_time=1.0,
        timing_breakdown={},
        success=True,
        refinement_count=1,
    )
    assert result_with_refinement.refinement_count == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_timing_breakdown_tracks_refinement_separately(
    mock_optimizer, low_quality_papers, high_quality_papers
):
    """Test that timing breakdown separately tracks initial and refinement stages."""
    service = QueryService(embedding_model=None)
    service.discovery_service._optimizer = mock_optimizer

    call_count = 0

    async def mock_search(queries, limit_per_query=100):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return low_quality_papers
        return high_quality_papers

    mock_client = AsyncMock()
    mock_client.multi_query_search = mock_search
    service.discovery_service._client = mock_client

    # Mock quality gate: fail first, pass second
    quality_call_count = 0

    def mock_validate_quality(papers, query_embedding):
        nonlocal quality_call_count
        quality_call_count += 1
        if quality_call_count == 1:
            return {
                "passed": False,
                "metrics": {
                    "paper_count": 2,
                    "embedding_coverage": 0.0,
                    "abstract_coverage": 1.0,
                },
                "reason": "insufficient paper count",
            }
        return {
            "passed": True,
            "metrics": {
                "paper_count": 30,
                "embedding_coverage": 0.8,
                "abstract_coverage": 1.0,
            },
            "reason": "",
        }

    with patch(
        "services.discovery_service.validate_quality", side_effect=mock_validate_quality
    ):
        result = await service.process_query(
            "What are the cardiovascular effects of intermittent fasting?"
        )

    # Should have both initial and refinement timings
    timing = result.timing_breakdown

    assert "semantic_scholar_search" in timing, "Should track initial search time"
    assert (
        "semantic_scholar_search_refinement" in timing
    ), "Should track refinement search time"
    assert "quality_gate" in timing, "Should track initial quality gate time"
    assert (
        "quality_gate_refinement" in timing
    ), "Should track refinement quality gate time"

    # All timings should be positive
    assert timing["semantic_scholar_search"] >= 0
    assert timing["semantic_scholar_search_refinement"] >= 0
    assert timing["quality_gate"] >= 0
    assert timing["quality_gate_refinement"] >= 0
