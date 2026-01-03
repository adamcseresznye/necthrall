import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.query_optimization_agent import QueryOptimizationAgent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_dual_queries_valid_query():
    """generate_dual_queries should return four distinct optimized queries for a valid input"""
    agent = QueryOptimizationAgent()
    query = "fasting risks"

    expected_output = {
        "intent_type": "general",
        "final_rephrase": "cardiovascular and metabolic risks associated with intermittent fasting protocols",
        "primary": "intermittent fasting cardiovascular risks adverse effects",
        "broad": "fasting protocols health outcomes safety cardiovascular metabolic",
        "alternative": "time-restricted eating cardiac complications health risks",
    }

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = json.dumps(expected_output)

        result = await agent.generate_dual_queries(query)

        assert result == expected_output
        assert all(isinstance(v, str) and v for v in result.values())
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_dual_queries_llm_timeout_fallback():
    """If LLM call fails, generate_dual_queries should return original query for all fields"""
    agent = QueryOptimizationAgent()
    query = "test query"

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = Exception("LLM timeout")

        result = await agent.generate_dual_queries(query)

        expected = {
            "strategy": "expansion",
            "final_rephrase": query,
            "primary": query,
            "broad": query,
            "alternative": query,
        }
        assert result == expected
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_dual_queries_invalid_json_fallback():
    """If LLM returns invalid JSON, generate_dual_queries should return original query for all fields"""
    agent = QueryOptimizationAgent()
    query = "test query"

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "invalid json {"

        result = await agent.generate_dual_queries(query)

        expected = {
            "strategy": "expansion",
            "final_rephrase": query,
            "primary": query,
            "broad": query,
            "alternative": query,
        }
        assert result == expected
        mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_dual_queries_empty_query():
    """generate_dual_queries should handle empty query gracefully by returning empty strings"""
    agent = QueryOptimizationAgent()
    query = ""

    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = json.dumps(
            {"final_rephrase": "", "primary": "", "broad": "", "alternative": ""}
        )

        result = await agent.generate_dual_queries(query)

        expected = {
            "intent_type": "general",
            "final_rephrase": "",
            "primary": "",
            "broad": "",
            "alternative": "",
        }
        assert result == expected
        mock_generate.assert_called_once()
        mock_generate.assert_called_once()
