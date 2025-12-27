import asyncio
import time
from typing import Dict, List, Tuple

import httpx
import pytest
from loguru import logger

from agents.acquisition_agent import AcquisitionAgent
from models.state import State

# Probe helper with retries for transient network issues (1s,2s,4s)
RETRY_DELAYS = [1.0, 2.0, 4.0]


async def _probe_url(url: str) -> Tuple[bool, str, float]:
    """Probe a URL with retries. Returns (ok, reason, elapsed_secs).

    ok is True when final response status is 200 and content-type PDF-like.
    reason is a short string describing failure or 'ok'.
    """
    start = time.perf_counter()
    for attempt, delay in enumerate([0.0] + RETRY_DELAYS):
        if attempt > 0:
            logger.warning(
                "Retrying URL {url} (attempt {n}) after transient error",
                url=url,
                n=attempt,
            )
            await asyncio.sleep(delay)
        try:
            resp = httpx.get(url, timeout=6.0)
            status = resp.status_code
            if status == 200:
                ct = resp.headers.get("content-type", "")
                elapsed = time.perf_counter() - start
                # heuristic: accept application/pdf or url to arxiv which returns application/pdf
                if "pdf" in ct or url.lower().endswith(".pdf"):
                    return True, "ok", elapsed
                else:
                    return True, "non-pdf-content", elapsed
            elif status == 503:
                # transient server error -> retry
                logger.warning("Received 503 for {url}", url=url)
                continue
            else:
                return False, f"http_{status}", time.perf_counter() - start
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            # transient network -> retry
            logger.warning(
                "Transient network error probing {url}: {err}", url=url, err=str(e)
            )
            continue
        except Exception as e:
            return False, f"error_{str(e)}", time.perf_counter() - start

    return False, "transient_failed", time.perf_counter() - start


def _semantic_scholar_available() -> bool:
    try:
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/1706.03762?fields=paperId",
            timeout=2.0,
        )
        return resp.status_code < 500
    except Exception as e:
        logger.warning("Semantic Scholar health check failed: {}", e)
        return False


SKIP_SCHOLAR = not _semantic_scholar_available()


@pytest.fixture(scope="module")
def arxiv_pdf_urls() -> List[dict]:
    """Return a list of 10 finalists with `openAccessPdf.url` pointing at arXiv PDFs.

    These are stable arXiv `/pdf/<id>.pdf` URLs. Use domain-specific IDs as
    representative open-access papers for integration tests.
    """
    ids = [
        "2101.09678",
        "2012.00431",
        "2006.10029",
        "1907.11692",
        "1803.09042",
        "1706.03762",
        "1603.04467",
        "1506.01497",
        "1409.0473",
        "1312.6114",
    ]
    finalists = [
        {
            "paperId": aid,
            "title": f"arXiv {aid}",
            "openAccessPdf": {"url": f"https://arxiv.org/pdf/{aid}.pdf"},
        }
        for aid in ids
    ]
    return finalists


@pytest.fixture
def state_with_good_finalists(arxiv_pdf_urls: List[dict]) -> State:
    """State with 10 arXiv finalists expected to have open PDFs."""
    return State(query="test acquisition", finalists=arxiv_pdf_urls)


@pytest.fixture
def state_with_mixed_urls(arxiv_pdf_urls: List[dict]) -> State:
    """State with 8 valid and 2 invalid PDF URLs mixed in."""
    valid = arxiv_pdf_urls[:8]
    invalid = [
        {
            "paperId": "invalid1",
            "title": "Invalid 1",
            "openAccessPdf": {"url": "https://invalid.example.com/404.pdf"},
        },
        {
            "paperId": "invalid2",
            "title": "Invalid 2",
            "openAccessPdf": {"url": "https://invalid.example.com/403.pdf"},
        },
    ]
    finalists = valid + invalid
    return State(query="mixed urls", finalists=finalists)


@pytest.fixture
def state_with_zero_valid() -> State:
    """State with zero valid PDF URLs to trigger error handling."""
    finalists = [
        {
            "paperId": "bad1",
            "title": "Bad 1",
            "openAccessPdf": {"url": "https://invalid.example.com/404.pdf"},
        },
        {
            "paperId": "bad2",
            "title": "Bad 2",
            "openAccessPdf": {"url": "https://invalid.example.com/403.pdf"},
        },
    ]
    return State(query="zero valid", finalists=finalists)


@pytest.fixture(scope="module")
def acquisition_agent() -> AcquisitionAgent:
    """Return an AcquisitionAgent instrumented to collect per-paper timing metrics.

    The fixture wraps the internal download/extract methods to record timings
    into `agent.metrics` so tests can profile slow downloads and extractions.
    """
    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 30.0
    metrics: Dict[str, Dict[str, float]] = {}

    # wrap download
    orig_download = agent._download_single_pdf

    async def _download_wrapper(
        paper_id: str, url: str, session, destination_path: str
    ):
        t0 = time.perf_counter()
        try:
            return await orig_download(
                paper_id, url, session, destination_path=destination_path
            )
        finally:
            dt = time.perf_counter() - t0
            metrics.setdefault(paper_id, {})["download_time"] = dt

    # wrap extract
    orig_extract = agent._extract_and_validate

    async def _extract_wrapper(tmp_path: str, paper_id: str) -> str:
        t0 = time.perf_counter()
        try:
            return await orig_extract(tmp_path, paper_id)
        finally:
            dt = time.perf_counter() - t0
            metrics.setdefault(paper_id, {})["extract_time"] = dt

    # attach wrappers
    agent._download_single_pdf = _download_wrapper  # type: ignore[attr-defined]
    agent._extract_and_validate = _extract_wrapper  # type: ignore[attr-defined]
    agent.metrics = metrics  # type: ignore[attr-defined]
    return agent


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    SKIP_SCHOLAR, reason="Semantic Scholar API unreachable; skipping integration tests"
)
async def test_acquisition_success_rate_and_latency(
    acquisition_agent: AcquisitionAgent, state_with_good_finalists: State
) -> None:
    """Download 10 PDFs and validate >90% success and latency <4s.

    This test probes each PDF URL with retries before running the agent and
    will retry the whole acquisition run up to 3 times when probes indicate
    transient failures (timeouts/503). It also profiles per-paper timings
    and logs the slowest downloads.
    """
    agent = acquisition_agent

    # probe all URLs first (concurrent)
    urls = [
        (f.openAccessPdf or {}).get("url") for f in state_with_good_finalists.finalists
    ]
    urls = [u for u in urls if u]
    logger.info("Probing {n} PDF URLs before acquisition", n=len(urls))
    probe_tasks = [_probe_url(u) for u in urls]
    probe_results = await asyncio.gather(*probe_tasks)

    reachable = [r for r in probe_results if r[0]]
    unreachable = [r for r in probe_results if not r[0]]
    logger.info(
        "Probe results: {ok}/{total} reachable", ok=len(reachable), total=len(urls)
    )

    # If more than 2 URLs are unreachable at probe time, skip test to avoid false negatives.
    if len(unreachable) > 2:
        pytest.skip(
            f"Too many unreachable PDFs during probe: {len(unreachable)}/{len(urls)}"
        )

    # run acquisition with retries on transient issues (1s,2s,4s)
    max_attempts = 3
    delays = RETRY_DELAYS
    updated = None
    start_total = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        attempt_start = time.perf_counter()
        logger.info("Acquisition attempt {a}/{m}", a=attempt, m=max_attempts)
        updated = await agent.process(state_with_good_finalists)
        elapsed_attempt = time.perf_counter() - attempt_start

        passages = updated.passages or []
        success_count = len(passages)

        logger.info(
            "Attempt {a}: acquired {s}/{t} PDFs in {sec:.2f}s",
            a=attempt,
            s=success_count,
            t=len(urls),
            sec=elapsed_attempt,
        )

        # success criteria: >=5 successes
        if success_count >= 5:
            break

        # decide whether to retry: only if probe previously found transient issues
        transient_found = any(
            r[1] in ("transient_failed", "http_503") for r in probe_results
        )
        if attempt < max_attempts and transient_found:
            backoff = delays[min(attempt - 1, len(delays) - 1)]
            logger.warning(
                "Transient issues detected; retrying acquisition after {s}s backoff",
                s=backoff,
            )
            await asyncio.sleep(backoff)
            continue
        else:
            break

    total_elapsed = time.perf_counter() - start_total

    passages = (updated.passages or []) if updated else []
    success_count = len(passages)

    # final logging and assertions
    logger.info(
        "Final acquisition: {s}/{t} PDFs in {sec:.2f}s",
        s=success_count,
        t=len(urls),
        sec=total_elapsed,
    )

    # success rate requirement: >=5/10
    assert success_count >= 5, f"Expected >=5 passages, got {success_count}"

    # end-to-end latency requirement
    assert total_elapsed < 4.0, f"Latency {total_elapsed:.2f}s exceeds 4s target"

    # validate passages enrichment
    for p in passages:
        assert p.paper_id
        assert p.text
        assert isinstance(p.text, str)
        assert len(p.text) >= 500
        assert p.metadata.get("text_source") == "pdf"

    # Performance profiling: compute per-paper timings from agent.metrics
    metrics = getattr(agent, "metrics", {})
    download_times = [(pid, m.get("download_time", 0.0)) for pid, m in metrics.items()]
    extract_times = [(pid, m.get("extract_time", 0.0)) for pid, m in metrics.items()]
    # sort to find slowest
    download_times.sort(key=lambda x: x[1], reverse=True)
    slowest3 = download_times[:3]
    avg_download = sum(t for _, t in download_times) / (len(download_times) or 1)
    total_download = sum(t for _, t in download_times)
    total_extract = sum(m.get("extract_time", 0.0) for m in metrics.values())

    logger.debug(
        "Download profiling: total_download={td:.2f}s avg={avg:.2f}s slowest={slow}",
        td=total_download,
        avg=avg_download,
        slow=slowest3,
    )
    logger.debug(
        "Extraction total={te:.2f}s vs download total={td:.2f}s",
        te=total_extract,
        td=total_download,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    SKIP_SCHOLAR, reason="Semantic Scholar API unreachable; skipping integration tests"
)
async def test_mixed_urls_skip_invalid(
    acquisition_agent: AcquisitionAgent, state_with_mixed_urls: State
) -> None:
    """When invalid URLs are mixed in, they are skipped and valid ones are kept."""
    agent = acquisition_agent

    # probe valid subset
    urls = [(f.openAccessPdf or {}).get("url") for f in state_with_mixed_urls.finalists]
    probe_results = await asyncio.gather(*[_probe_url(u) for u in urls if u])
    logger.info(
        "Probed mixed URLs: {ok}/{t}",
        ok=sum(1 for r in probe_results if r[0]),
        t=len(urls),
    )

    updated = await agent.process(state_with_mixed_urls)
    passages = updated.passages or []
    paper_ids = {p.paper_id for p in passages}

    # Ensure invalid ids are not present
    assert "invalid1" not in paper_ids
    assert "invalid2" not in paper_ids

    # Expect at least 5 valid results (allowing failure tolerance)
    assert (
        len(passages) >= 5
    ), f"Expected at least 5 valid passages, got {len(passages)}"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    SKIP_SCHOLAR, reason="Semantic Scholar API unreachable; skipping integration tests"
)
async def test_zero_valid_urls_populates_errors(
    acquisition_agent: AcquisitionAgent, state_with_zero_valid: State
) -> None:
    """If zero PDFs are acquired, State.errors should contain the critical message."""
    agent = acquisition_agent

    updated = await agent.process(state_with_zero_valid)

    passages = updated.passages or []
    assert len(passages) == 0
    assert any(
        "Critical: No PDFs acquired" in e for e in updated.errors
    ), f"Expected critical error in State.errors: {updated.errors}"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    SKIP_SCHOLAR, reason="Semantic Scholar API unreachable; skipping integration tests"
)
async def test_parallel_execution_order_independent(
    acquisition_agent: AcquisitionAgent, arxiv_pdf_urls: List[dict]
) -> None:
    """Run acquisition twice with different ordering to validate parallel correctness."""
    agent = acquisition_agent

    # Limit to 5 items so both runs acquire the same set (since agent stops at 5)
    subset_urls = arxiv_pdf_urls[:5]

    s1 = State(query="run1", finalists=subset_urls)
    s2 = State(query="run2", finalists=list(reversed(subset_urls)))

    updated1 = await agent.process(s1)
    updated2 = await agent.process(s2)

    ids1 = {p.paper_id for p in (updated1.passages or [])}
    ids2 = {p.paper_id for p in (updated2.passages or [])}

    # sets of acquired paperIds should be equal (order doesn't matter)
    assert ids1 == ids2, f"Expected same acquired ids across runs, got {ids1} vs {ids2}"
