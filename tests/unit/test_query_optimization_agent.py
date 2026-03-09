"""Unit tests for QueryOptimizationAgent.optimize() method."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.query_optimization_agent import QueryOptimizationAgent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_optimize_returns_two_fields():
    """optimize() should return dict with intent_type and final_rephrase."""
    agent = QueryOptimizationAgent()
    query = "fasting risks"

    expected_output = {
        "intent_type": "general",
        "final_rephrase": "cardiovascular and metabolic risks of intermittent fasting",
    }

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = json.dumps(expected_output)

        result = await agent.optimize(query)

        assert result == expected_output
        assert "intent_type" in result
        assert "final_rephrase" in result
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_optimize_llm_failure_fallback():
    """If LLM call fails, optimize() should return fallback with original query."""
    agent = QueryOptimizationAgent()
    query = "test query"

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = Exception("LLM timeout")

        result = await agent.optimize(query)

        assert result["intent_type"] == "general"
        assert result["final_rephrase"] == query
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_optimize_invalid_json_fallback():
    """If LLM returns invalid JSON, optimize() should return fallback."""
    agent = QueryOptimizationAgent()
    query = "test query"

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "invalid json {"

        result = await agent.optimize(query)

        assert result["intent_type"] == "general"
        assert result["final_rephrase"] == query
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_optimize_empty_query():
    """optimize() should handle empty query gracefully."""
    agent = QueryOptimizationAgent()
    query = ""

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = json.dumps(
            {"intent_type": "general", "final_rephrase": ""}
        )

        result = await agent.optimize(query)

        assert result["intent_type"] == "general"
        assert result["final_rephrase"] == ""
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_optimize_preserves_intent_type():
    """optimize() should preserve intent_type from LLM response."""
    agent = QueryOptimizationAgent()
    query = "what are seminal works on CRISPR?"

    expected_output = {
        "intent_type": "foundational",
        "final_rephrase": "CRISPR-Cas9 seminal works review gene editing",
    }

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = json.dumps(expected_output)

        result = await agent.optimize(query)

        assert result["intent_type"] == "foundational"
        assert result["final_rephrase"] == expected_output["final_rephrase"]
