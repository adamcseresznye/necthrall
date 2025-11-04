import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from loguru import logger

from agents.analysis import ContradictionDetector, LLMProviderError


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_contradiction_detector_parses_valid_response(
    valid_llm_response_json, contradictory_passages, langgraph_state_factory
):
    detector = ContradictionDetector()

    # Patch internal LLM call to return our valid JSON
    detector._call_llm_with_fallback = AsyncMock(return_value=valid_llm_response_json)

    state = langgraph_state_factory(
        filtered_papers=None, relevant_passages=contradictory_passages
    )

    results = await detector.detect_contradictions(
        query=state.original_query,
        relevant_passages=[p.model_dump() for p in contradictory_passages],
        llm_config={},
    )

    assert isinstance(results, list)
    assert len(results) >= 1
    first = results[0]
    assert first.topic.startswith("effect_of_drug") or hasattr(first, "topic")


@pytest.mark.asyncio
async def test_contradiction_detector_handles_malformed_json(
    malformed_llm_response, contradictory_passages
):
    detector = ContradictionDetector()
    detector._call_llm_with_fallback = AsyncMock(return_value=malformed_llm_response)

    # Replace logger.warning with mock to capture parse warnings
    with patch.object(logger, "warning", Mock()) as mock_warn:
        results = await detector.detect_contradictions(
            query="q",
            relevant_passages=[p.model_dump() for p in contradictory_passages],
            llm_config={},
        )
        assert results == []
        assert mock_warn.called


@pytest.mark.asyncio
async def test_contradiction_detector_provider_failure_and_fallback(
    contradictory_passages,
):
    detector = ContradictionDetector()

    # Simulate LLM provider raising an error (e.g., timeout) so detect_contradictions returns [] and logs
    async def raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError("timeout")

    detector._call_llm_with_fallback = AsyncMock(side_effect=raise_timeout)

    with patch.object(logger, "error", Mock()) as mock_err:
        results = await detector.detect_contradictions(
            query="q",
            relevant_passages=[p.model_dump() for p in contradictory_passages],
            llm_config={},
        )
        # On provider failure we expect empty list (detect_contradictions catches and returns [])
        assert results == []
        assert mock_err.called
