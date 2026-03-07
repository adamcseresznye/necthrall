from unittest.mock import AsyncMock, MagicMock

import pytest

from services.research_service import MAX_ROUNDS, ResearchService


def _make_paper(paper_id: str):
    p = MagicMock()
    p.paperId = paper_id
    return p


def _make_discovery(paper_ids: list[str]):
    result = MagicMock()
    result.finalists = [_make_paper(pid) for pid in paper_ids]
    return result


def _make_ingestion(n_chunks: int):
    result = MagicMock()
    result.chunks = [MagicMock() for _ in range(n_chunks)]
    return result


def _make_rag(answer: str | None):
    result = MagicMock()
    result.answer = answer
    return result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_research_exits_round1_when_complete():
    """Reflection says is_complete=True on round 1 → only 1 round fires."""
    discovery = AsyncMock()
    discovery.discover.return_value = _make_discovery(["p1", "p2", "p3"])
    ingestion = AsyncMock()
    ingestion.ingest.return_value = _make_ingestion(10)
    rag = AsyncMock()
    rag.answer.return_value = _make_rag("Some answer")

    planning_result = {
        "research_brief": "Brief about topic X. " * 10,
        "initial_query": "topic X overview",
        "final_rephrase": "What is topic X?",
    }
    reflection_result = {"is_complete": True, "gap_description": "", "next_query": None}

    svc = ResearchService(discovery, ingestion, rag)
    svc._planning_agent.plan = AsyncMock(return_value=planning_result)
    svc._reflection_agent.evaluate = AsyncMock(return_value=reflection_result)

    result = await svc.research("What is topic X?")

    assert result.rounds_completed == 1
    assert result.answer == "Some answer"
    discovery.discover.assert_called_once_with("topic X overview")
    # final_rephrase used for synthesis, not gap query
    rag.answer.assert_called_once()
    call_query = rag.answer.call_args[0][0]
    assert call_query == "What is topic X?"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_seen_paper_ids_prevent_duplicates():
    """Papers returned in round 2 that were already seen in round 1 are filtered out."""
    discovery = AsyncMock()
    # Round 1 returns p1, p2, p3. Round 2 returns p1 (duplicate) and p4 (new).
    discovery.discover.side_effect = [
        _make_discovery(["p1", "p2", "p3"]),
        _make_discovery(["p1", "p4"]),  # p1 already seen
    ]
    ingestion = AsyncMock()
    ingestion.ingest.return_value = _make_ingestion(5)
    rag = AsyncMock()
    rag.answer.return_value = _make_rag("Answer")

    round1_reflection = {
        "is_complete": False,
        "gap_description": "missing aspect Y",
        "next_query": "topic Y details",
    }
    round2_reflection = {"is_complete": True, "gap_description": "", "next_query": None}

    svc = ResearchService(discovery, ingestion, rag)
    svc._planning_agent.plan = AsyncMock(
        return_value={
            "research_brief": "B " * 50,
            "initial_query": "topic X",
            "final_rephrase": "What is X?",
        }
    )
    svc._reflection_agent.evaluate = AsyncMock(
        side_effect=[round1_reflection, round2_reflection]
    )

    result = await svc.research("What is X?")

    assert result.rounds_completed == 2
    # Round 2 ingest should only receive p4 (not p1)
    round2_call_finalists = ingestion.ingest.call_args_list[1].args[0]
    assert all(p.paperId != "p1" for p in round2_call_finalists)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_zero_new_papers_breaks_loop():
    """If filtered finalists is empty, loop breaks without calling ingest."""
    discovery = AsyncMock()
    discovery.discover.side_effect = [
        _make_discovery(["p1"]),
        _make_discovery(["p1"]),  # all already seen
    ]
    ingestion = AsyncMock()
    ingestion.ingest.return_value = _make_ingestion(5)
    rag = AsyncMock()
    rag.answer.return_value = _make_rag("Answer R1")

    svc = ResearchService(discovery, ingestion, rag)
    svc._planning_agent.plan = AsyncMock(
        return_value={
            "research_brief": "B " * 50,
            "initial_query": "topic X",
            "final_rephrase": "What is X?",
        }
    )
    svc._reflection_agent.evaluate = AsyncMock(
        return_value={
            "is_complete": False,
            "gap_description": "gap",
            "next_query": "follow-up",
        }
    )

    result = await svc.research("What is X?")

    # Round 2 discovery returns only p1 (seen) → breaks before ingest
    assert ingestion.ingest.call_count == 1
    assert result.answer == "Answer R1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_synthesis_none_preserves_previous_answer():
    """If synthesis returns None in round 2, the round 1 answer is preserved."""
    discovery = AsyncMock()
    discovery.discover.side_effect = [
        _make_discovery(["p1", "p2"]),
        _make_discovery(["p3"]),
    ]
    ingestion = AsyncMock()
    ingestion.ingest.return_value = _make_ingestion(5)
    rag = AsyncMock()
    rag.answer.side_effect = [_make_rag("Round 1 answer"), _make_rag(None)]

    svc = ResearchService(discovery, ingestion, rag)
    svc._planning_agent.plan = AsyncMock(
        return_value={
            "research_brief": "B " * 50,
            "initial_query": "topic X",
            "final_rephrase": "What is X?",
        }
    )
    svc._reflection_agent.evaluate = AsyncMock(
        return_value={"is_complete": True, "gap_description": "", "next_query": None}
    )

    result = await svc.research("What is X?")
    assert result.answer == "Round 1 answer"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_corpus_grows_across_rounds():
    """current_chunks grows (merge) not resets each round."""
    discovery = AsyncMock()
    discovery.discover.side_effect = [
        _make_discovery(["p1"]),
        _make_discovery(["p2"]),
    ]
    ingestion = AsyncMock()
    ingestion.ingest.side_effect = [
        _make_ingestion(10),  # round 1: 10 chunks
        _make_ingestion(5),  # round 2: 5 chunks
    ]
    rag = AsyncMock()
    rag.answer.return_value = _make_rag("Answer")

    round1_ref = {"is_complete": False, "gap_description": "g", "next_query": "q2"}
    round2_ref = {"is_complete": True, "gap_description": "", "next_query": None}

    svc = ResearchService(discovery, ingestion, rag)
    svc._planning_agent.plan = AsyncMock(
        return_value={
            "research_brief": "B " * 50,
            "initial_query": "topic X",
            "final_rephrase": "What is X?",
        }
    )
    svc._reflection_agent.evaluate = AsyncMock(side_effect=[round1_ref, round2_ref])

    result = await svc.research("What is X?")

    assert result.total_chunks == 15  # 10 + 5 merged
    # Round 2 RAG call should have 15 chunks (not just 5)
    round2_rag_chunks = rag.answer.call_args_list[1].args[1]
    assert len(round2_rag_chunks) == 15
    round2_rag_chunks = rag.answer.call_args_list[1].args[1]
    assert len(round2_rag_chunks) == 15
