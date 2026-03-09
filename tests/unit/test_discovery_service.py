"""Unit tests for DiscoveryService — dedup filter and top_k changes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.config import Settings
from models.state import Paper
from services.discovery_service import DiscoveryResult, DiscoveryService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.SEMANTIC_SCHOLAR_API_KEY = "fake-key"
    return settings


def _make_paper(paper_id: str, title: str = "") -> Paper:
    """Return a minimal Paper object."""
    return Paper(paperId=paper_id, title=title or f"Paper {paper_id}")


def _make_discovery_service(mock_settings) -> DiscoveryService:
    """Construct DiscoveryService with all heavy dependencies stubbed out."""
    with (
        patch("services.discovery_service.QueryOptimizationAgent"),
        patch("services.discovery_service.SemanticScholarClient"),
        patch("services.discovery_service.RankingAgent"),
    ):
        svc = DiscoveryService(mock_settings)
    return svc


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _patch_search_and_gate(svc: DiscoveryService, papers: list):
    """Patch _execute_search_and_quality_gate to return given papers + passing gate."""
    svc._execute_search_and_quality_gate = AsyncMock(
        return_value=(
            [p.__dict__ for p in papers],  # raw dicts as returned by the search layer
            {"passed": True},
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discover_dedup_filters_seen_papers(mock_settings):
    """discover(..., existing_paper_ids={'aaa'}) must NOT return a finalist with paperId='aaa'."""
    svc = _make_discovery_service(mock_settings)

    paper_a = _make_paper("aaa", "Paper A")
    paper_b = _make_paper("bbb", "Paper B")

    # optimize returns a dict with intent_type and final_rephrase
    svc.optimizer.optimize = AsyncMock(
        return_value={"intent_type": "general", "final_rephrase": "test query keyword"}
    )

    # rank_papers returns both papers synchronously; we bypass asyncio.to_thread
    svc.ranker.rank_papers = MagicMock(return_value=[paper_a, paper_b])

    _patch_search_and_gate(svc, [paper_a, paper_b])

    result: DiscoveryResult = await svc.discover(
        "test query", existing_paper_ids={"aaa"}
    )

    ids = [p.paperId for p in result.finalists]
    assert "aaa" not in ids, "Seen paper 'aaa' should have been filtered out"
    assert "bbb" in ids, "Unseen paper 'bbb' should still be present"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discover_no_dedup_without_existing_ids(mock_settings):
    """discover(query) with no existing_paper_ids returns all finalists unfiltered."""
    svc = _make_discovery_service(mock_settings)

    paper_a = _make_paper("aaa", "Paper A")
    paper_b = _make_paper("bbb", "Paper B")

    svc.optimizer.optimize = AsyncMock(
        return_value={"intent_type": "general", "final_rephrase": "test query keyword"}
    )
    svc.ranker.rank_papers = MagicMock(return_value=[paper_a, paper_b])

    _patch_search_and_gate(svc, [paper_a, paper_b])

    result: DiscoveryResult = await svc.discover("test query")

    ids = [p.paperId for p in result.finalists]
    assert "aaa" in ids
    assert "bbb" in ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discover_uses_optimize_always(mock_settings):
    """DiscoveryService always uses optimizer.optimize() for query optimization."""
    svc = _make_discovery_service(mock_settings)

    svc.optimizer.optimize = AsyncMock(
        return_value={"intent_type": "general", "final_rephrase": "optimized query"}
    )
    svc.ranker.rank_papers = MagicMock(return_value=[])

    _patch_search_and_gate(svc, [])

    await svc.discover("test query", existing_paper_ids=set())

    svc.optimizer.optimize.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discover_dedup_empty_set_does_not_filter(mock_settings):
    """An empty existing_paper_ids set should not filter any finalists."""
    svc = _make_discovery_service(mock_settings)

    paper_a = _make_paper("aaa", "Paper A")

    svc.optimizer.optimize = AsyncMock(
        return_value={"intent_type": "general", "final_rephrase": "keyword query"}
    )
    svc.ranker.rank_papers = MagicMock(return_value=[paper_a])

    _patch_search_and_gate(svc, [paper_a])

    result: DiscoveryResult = await svc.discover("test query", existing_paper_ids=set())

    # Empty set — falsy — so dedup block is skipped; all finalists returned
    ids = [p.paperId for p in result.finalists]
    assert "aaa" in ids
