from unittest.mock import AsyncMock, patch

import pytest

from agents.planning_agent import PlanningAgent

DIVERSE_QUERIES = [
    "what is the mechanism of CRISPR-Cas9 gene editing?",
    "cognitive effects of chronic sleep deprivation in adults",
    "microbiome effects on Parkinson's disease",
    "SGLT2 inhibitors cardiovascular outcomes type 2 diabetes",
    "mTOR pathway aging longevity interventions",
]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_returns_required_keys():
    agent = PlanningAgent()
    mock_json = (
        '{"research_brief": "' + ("word " * 110).strip() + '",'
        '"initial_query": "CRISPR Cas9 gene editing mechanism",'
        '"final_rephrase": "What is the molecular mechanism of CRISPR-Cas9?"}'
    )
    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_json
        result = await agent.plan("what is the mechanism of CRISPR-Cas9?")

    assert set(result.keys()) == {"research_brief", "initial_query", "final_rephrase"}


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("query", DIVERSE_QUERIES)
async def test_research_brief_is_substantive(query):
    """research_brief must be >100 words on diverse queries."""
    agent = PlanningAgent()
    mock_json = (
        '{"research_brief": "' + ("detailed word " * 80).strip() + '",'
        '"initial_query": "keyword search terms",'
        '"final_rephrase": "' + query + '"}'
    )
    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_json
        result = await agent.plan(query)

    word_count = len(result["research_brief"].split())
    assert word_count > 100, f"Brief too short: {word_count} words"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initial_query_has_no_question_mark():
    agent = PlanningAgent()
    mock_json = (
        '{"research_brief": "' + ("word " * 110).strip() + '",'
        '"initial_query": "sleep deprivation cognitive effects adults",'
        '"final_rephrase": "What are the cognitive effects of chronic sleep deprivation?"}'
    )
    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_json
        result = await agent.plan("cognitive effects of chronic sleep deprivation")

    assert "?" not in result["initial_query"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_fires_on_llm_exception():
    """All three fields must equal the original query when LLM raises."""
    agent = PlanningAgent()
    query = "some research question"
    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = RuntimeError("LLM down")
        result = await agent.plan(query)

    assert result["research_brief"] == query
    assert result["initial_query"] == query
    assert result["final_rephrase"] == query


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_fires_on_invalid_json():
    """Fallback fires when LLM returns unparseable output."""
    agent = PlanningAgent()
    query = "some research question"
    with patch.object(
        agent.router, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "not valid json at all"
        result = await agent.plan(query)

    assert result["research_brief"] == query
