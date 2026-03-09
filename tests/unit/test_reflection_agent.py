from unittest.mock import AsyncMock, patch

import pytest

from agents.reflection_agent import ReflectionAgent, ReflectionResult

BRIEF = (
    "A complete answer must cover: (1) RCT evidence for efficacy in adults, "
    "(2) mechanism of action via mTOR inhibition, (3) side-effect profiles "
    "from phase III trials, (4) comparison with caloric restriction in animal models."
)
GOOD_ANSWER = (
    "Rapamycin extends lifespan via mTOR inhibition, consistent with caloric "
    "restriction mimicry. RCTs in adults show efficacy. Side effects from phase III "
    "include immunosuppression. Animal models confirm the mechanism."
)
POOR_ANSWER = "Rapamycin is a drug used in transplant medicine."

PRIOR_QUERIES = ["rapamycin mTOR lifespan", "rapamycin clinical trials"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_answer_returns_is_complete_true():
    agent = ReflectionAgent()
    llm_response = '{"is_complete": true, "gap_description": "", "next_query": null}'
    with patch.object(agent, "_call_llm", new=AsyncMock(return_value=llm_response)):
        result = await agent.evaluate("rapamycin query", BRIEF, GOOD_ANSWER, PRIOR_QUERIES)
    assert result["is_complete"] is True
    assert result["next_query"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incomplete_answer_returns_gap_and_query():
    agent = ReflectionAgent()
    llm_response = (
        '{"is_complete": false, '
        '"gap_description": "Missing side-effect profiles from phase III trials.", '
        '"next_query": "rapamycin phase III side effects"}'
    )
    with patch.object(agent, "_call_llm", new=AsyncMock(return_value=llm_response)):
        result = await agent.evaluate("rapamycin query", BRIEF, POOR_ANSWER, PRIOR_QUERIES)
    assert result["is_complete"] is False
    assert result["next_query"] == "rapamycin phase III side effects"
    assert "side-effect" in result["gap_description"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_failure_returns_safe_fallback():
    agent = ReflectionAgent()
    with patch.object(agent, "_call_llm", new=AsyncMock(return_value=None)):
        result = await agent.evaluate("any query", BRIEF, POOR_ANSWER, PRIOR_QUERIES)
    assert result["is_complete"] is True
    assert result["next_query"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_duplicate_next_query_forces_complete():
    agent = ReflectionAgent()
    # next_query is a duplicate of an already-searched query
    llm_response = (
        '{"is_complete": false, '
        '"gap_description": "Missing RCT data.", '
        '"next_query": "rapamycin mTOR lifespan"}'  # already in PRIOR_QUERIES
    )
    with patch.object(agent, "_call_llm", new=AsyncMock(return_value=llm_response)):
        result = await agent.evaluate("rapamycin query", BRIEF, POOR_ANSWER, PRIOR_QUERIES)
    assert result["is_complete"] is True  # duplicate detected → safe fallback


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_json_returns_safe_fallback():
    agent = ReflectionAgent()
    with patch.object(agent, "_call_llm", new=AsyncMock(return_value="not json at all")):
        result = await agent.evaluate("any query", BRIEF, POOR_ANSWER, PRIOR_QUERIES)
    assert result["is_complete"] is True
