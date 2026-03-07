"""Unit tests for QueryService delegation to ResearchService."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.query_service import PipelineResult, QueryService
from services.research_service import ResearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Minimal Settings mock — prevents real network/model initialisation."""
    settings = MagicMock()
    settings.semantic_scholar_api_key = "fake-key"
    settings.primary_llm_api_key = "fake-key"
    return settings


@pytest.fixture
def mock_embedding_model():
    """Embedding model stub — returns a zero vector for any input."""
    model = MagicMock()
    model.encode = MagicMock(return_value=[0.0] * 384)
    return model


@pytest.fixture
def query_service(mock_settings, mock_embedding_model):
    """QueryService with all sub-service constructors patched to no-ops."""
    with (
        patch("services.query_service.QueryOptimizationAgent"),
        patch("services.query_service.DiscoveryService"),
        patch("services.query_service.IngestionService"),
        patch("services.query_service.RAGService"),
        patch("services.query_service.ResearchService"),
    ):
        svc = QueryService(mock_settings, mock_embedding_model)
    return svc


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_delegates_to_research_service(query_service):
    """process_query should delegate to research_service.research() and map fields."""
    fake_result = ResearchResult(
        answer="The answer is X.",
        research_brief="Brief about X.",
        rounds_completed=2,
        searched_queries=["query1", "query2"],
        total_chunks=15,
        timing_breakdown={"total": 12.5},
    )
    query_service.research_service.research = AsyncMock(return_value=fake_result)

    result = await query_service.process_query("What is X?")

    query_service.research_service.research.assert_awaited_once_with("What is X?")
    assert result.success is True
    assert result.answer == "The answer is X."
    assert result.refinement_count == 2
    assert result.optimized_queries["research_brief"] == "Brief about X."
    assert result.optimized_queries["searched_queries"] == ["query1", "query2"]
    assert result.timing_breakdown == {"total": 12.5}
    assert isinstance(result, PipelineResult)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_success_false_when_no_answer(query_service):
    """result.success must be False when research_result.answer is None."""
    fake_result = ResearchResult(
        answer=None,
        research_brief="",
        rounds_completed=1,
        searched_queries=[],
        total_chunks=0,
        timing_breakdown={},
    )
    query_service.research_service.research = AsyncMock(return_value=fake_result)

    result = await query_service.process_query("Unanswerable query")

    assert result.success is False
    assert result.answer is None
    assert result.refinement_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_exception_returns_error_result(query_service):
    """Any exception from research_service should be caught and surfaced in result."""
    query_service.research_service.research = AsyncMock(
        side_effect=RuntimeError("boom")
    )

    result = await query_service.process_query("Will explode")

    assert result.success is False
    assert result.error_stage == "research_service"
    assert "boom" in result.error_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_progress_callback_invoked(query_service):
    """progress_callback should be called before research_service.research()."""
    fake_result = ResearchResult(
        answer="ok",
        research_brief="brief",
        rounds_completed=1,
        searched_queries=[],
        total_chunks=0,
        timing_breakdown={},
    )
    query_service.research_service.research = AsyncMock(return_value=fake_result)

    callback = MagicMock()
    await query_service.process_query("test query", progress_callback=callback)

    callback.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_async_progress_callback_invoked(query_service):
    """Async progress_callback should be awaited correctly."""
    fake_result = ResearchResult(
        answer="ok",
        research_brief="brief",
        rounds_completed=1,
        searched_queries=[],
        total_chunks=0,
        timing_breakdown={},
    )
    query_service.research_service.research = AsyncMock(return_value=fake_result)

    async_callback = AsyncMock()
    await query_service.process_query("test query", progress_callback=async_callback)

    async_callback.assert_awaited_once()


@pytest.mark.unit
def test_query_service_has_research_service_attribute(query_service):
    """QueryService.__init__ must create a research_service attribute."""
    assert hasattr(query_service, "research_service")


@pytest.mark.unit
def test_pipeline_result_dataclass_fields():
    """PipelineResult must retain all expected fields unchanged."""
    result = PipelineResult(
        query="q",
        optimized_queries={},
        quality_gate={},
        finalists=[],
        execution_time=0.0,
        timing_breakdown={},
        success=True,
    )
    # Verify all fields from the original dataclass are present
    assert result.query == "q"
    assert result.optimized_queries == {}
    assert result.quality_gate == {}
    assert result.finalists == []
    assert result.execution_time == 0.0
    assert result.timing_breakdown == {}
    assert result.success is True
    assert result.error_message is None
    assert result.error_stage is None
    assert result.passages == []
    assert result.answer is None
    assert result.citation_verification is None
    assert result.refinement_count == 0
