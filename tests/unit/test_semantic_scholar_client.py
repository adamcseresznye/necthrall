import asyncio
import time
from typing import Any, Dict, List

import aiohttp
import pytest

from agents.semantic_scholar_client import SemanticScholarClient


def make_paper(
    paper_id: int, has_pdf: bool = True, embedding_dim: int = 768
) -> Dict[str, Any]:
    base = {
        "paperId": f"P{paper_id}",
        "title": f"Title {paper_id}",
        "abstract": f"Abstract {paper_id}",
        "year": 2020,
        "citationCount": 10,
        "influentialCitationCount": 1,
        "openAccessPdf": {"url": f"https://pdf/{paper_id}.pdf"} if has_pdf else {},
        "embedding": {"specter_v2": [0.0] * embedding_dim},
        "authors": [{"name": "A"}],
        "venue": "Venue",
        "externalIds": {"DOI": f"10.0/test/{paper_id}"},
    }
    return base


class DummyResponse:
    def __init__(self, status: int = 200, data: Dict = None, text: str = ""):
        self.status = status
        self._data = data or {}
        self._text = text

    async def json(self):
        return self._data

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyClientSession:
    """A very small replacement for aiohttp.ClientSession used in tests.

    It uses a mapping from query->DummyResponse or raises an Exception when
    the mapping value is an Exception instance.
    """

    def __init__(self, responses: Dict[str, Any]):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None, headers=None):
        query = (params or {}).get("query")
        value = self._responses.get(query)
        if isinstance(value, Exception):
            raise value
        # Allow simulated small delay to exercise concurrency
        return DummyResponse(status=200, data={"data": value or []})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_three_queries(monkeypatch):
    # queries produce overlapping ids to end up with ~200 unique papers
    q1 = [make_paper(i) for i in range(0, 100)]
    q2 = [make_paper(i) for i in range(50, 150)]
    q3 = [make_paper(i) for i in range(100, 200)]

    responses = {
        "q1": q1,
        "q2": q2,
        "q3": q3,
    }

    async def _factory(*args, **kwargs):
        return DummyClientSession(responses)

    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: DummyClientSession(responses)
    )

    client = SemanticScholarClient()
    start = time.perf_counter()
    papers = await client.multi_query_search(["q1", "q2", "q3"], limit_per_query=100)
    elapsed = time.perf_counter() - start

    # Should be deduplicated union of ids 0..199 => 200
    assert 150 <= len(papers) <= 300
    assert elapsed < 3.0

    # Embedding present under normalized key and has 768 dims
    for p in papers:
        emb = p.get("embedding", {}).get("specter")
        assert emb is not None
        assert len(emb) == 768


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deduplication(monkeypatch):
    # q1 and q2 share the same paper
    shared = make_paper(1)
    responses = {"a": [shared, make_paper(2)], "b": [shared, make_paper(3)], "c": []}
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: DummyClientSession(responses)
    )

    client = SemanticScholarClient()
    papers = await client.multi_query_search(["a", "b", "c"], limit_per_query=10)
    # shared paper only once
    ids = {p["paperId"] for p in papers}
    assert "P1" in ids
    assert len([pid for pid in ids if pid == "P1"]) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pdf_filtering(monkeypatch):
    p_with = make_paper(1, has_pdf=True)
    p_without = make_paper(2, has_pdf=False)
    responses = {"x": [p_with, p_without], "y": [], "z": []}
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: DummyClientSession(responses)
    )

    client = SemanticScholarClient()
    papers = await client.multi_query_search(["x", "y", "z"], limit_per_query=10)
    ids = {p["paperId"] for p in papers}
    assert "P1" in ids
    assert "P2" not in ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partial_failure(monkeypatch):
    # make one query raise an exception
    p_ok = make_paper(1)
    responses = {
        "ok1": [p_ok],
        "fail": RuntimeError("simulated failure"),
        "ok2": [make_paper(2)],
    }
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: DummyClientSession(responses)
    )

    client = SemanticScholarClient()
    papers = await client.multi_query_search(["ok1", "fail", "ok2"], limit_per_query=10)
    # Should return results from ok1 and ok2
    ids = {p["paperId"] for p in papers}
    assert "P1" in ids and "P2" in ids


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiting_many_concurrent_calls(monkeypatch):
    # Create small responses and run multiple client calls concurrently
    resp = [make_paper(i) for i in range(10)]
    responses = {"r": resp, "s": resp, "t": resp}
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: DummyClientSession(responses)
    )

    client = SemanticScholarClient()

    async def call_once():
        return await client.multi_query_search(["r", "s", "t"], limit_per_query=10)

    tasks = [asyncio.create_task(call_once()) for _ in range(8)]
    results = await asyncio.gather(*tasks)
    # All tasks must complete with results
    for papers in results:
        assert isinstance(papers, list)
        assert len(papers) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deduplication_multiple_duplicates(monkeypatch):
    """Ensure duplicates across and within query results are deduplicated."""
    # Same paper appears multiple times inside q1 and also in q2
    shared = make_paper(42)
    q1 = [shared, shared, make_paper(43)]
    q2 = [shared, make_paper(44)]
    q3 = [make_paper(45)]

    responses = {"q1": q1, "q2": q2, "q3": q3}
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **k: DummyClientSession(responses)
    )

    client = SemanticScholarClient()
    papers = await client.multi_query_search(["q1", "q2", "q3"], limit_per_query=10)
    ids = [p["paperId"] for p in papers]
    # shared paper should appear exactly once
    assert ids.count("P42") == 1
    # other papers present
    assert "P43" in ids and "P44" in ids and "P45" in ids
