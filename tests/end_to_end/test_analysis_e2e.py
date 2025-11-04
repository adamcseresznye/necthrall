import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from loguru import logger

from agents.analysis import AnalysisAgent

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_analysis_agent_end_to_end_success(
    langgraph_state_factory,
    high_cred_paper,
    contradictory_passages,
    valid_llm_response_json,
):
    agent = AnalysisAgent()

    # Create a state with one paper and passages
    state = langgraph_state_factory(
        filtered_papers=[high_cred_paper], relevant_passages=contradictory_passages
    )

    # Patch contradiction detector LLM call to return a valid JSON string
    agent.contradiction_detector._call_llm_with_fallback = AsyncMock(
        return_value=valid_llm_response_json
    )

    start = time.time()
    result = await agent.analyze(state)
    elapsed = time.time() - start

    # Basic contract: returns credibility_scores and contradictions
    assert "credibility_scores" in result
    assert "contradictions" in result
    assert len(result["credibility_scores"]) == 1
    # Execution time expected under 8s for e2e test (mocked LLM)
    assert result["execution_times"]["analysis_agent"] < 8.0
    assert elapsed < 8.0


@pytest.mark.asyncio
async def test_analysis_agent_partial_failure_credibility_ok_contradiction_fails(
    langgraph_state_factory, high_cred_paper, contradictory_passages
):
    agent = AnalysisAgent()
    state = langgraph_state_factory(
        filtered_papers=[high_cred_paper], relevant_passages=contradictory_passages
    )

    # Make contradiction detection raise to simulate provider failure
    async def raise_err(*args, **kwargs):
        raise RuntimeError("LLM provider down")

    agent.contradiction_detector.detect_contradictions = AsyncMock(
        side_effect=raise_err
    )

    # Patch logger.error to capture that an error was logged
    with patch.object(logger, "error", Mock()) as mock_err:
        result = await agent.analyze(state)
        # Credibility should still be present
        assert len(result["credibility_scores"]) == 1
        # Contradictions should be empty list on failure
        assert result["contradictions"] == []
        # Error should have been logged
        assert mock_err.called
