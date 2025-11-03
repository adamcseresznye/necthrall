#!/usr/bin/env python3
"""Compatibility shim for tests that expect a small performance test API.

This module provides a minimal set of symbols used by tests:
- NUM_PAPERS
- TEST_ITERATIONS
- TIMEOUT_SECONDS
- async def process_paper(session, url, timeout)
- async def main(...)
- PerformanceReport (imported from the more comprehensive validator)

The real, comprehensive validation lives in
`scripts/performance_validation.py` and is used for heavier validation runs.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import aiohttp

# Reuse the richer PerformanceReport definition from the validation script
try:
    from scripts.performance_validation import PerformanceReport
except Exception:
    # Fallback simple dataclass if import fails
    from dataclasses import dataclass

    @dataclass
    class PerformanceReport:
        total_time: float = 0.0
        avg_time: float = 0.0
        success: bool = True


# Default parameters used by tests
NUM_PAPERS = 25
TEST_ITERATIONS = 3
TIMEOUT_SECONDS = 30


async def process_paper(
    session: aiohttp.ClientSession, url: str, timeout: int
) -> Dict[str, Any]:
    """Fetch a PDF URL (or any URL) and return timing/info used by tests.

    Keeps behavior simple and robust: returns a dict with status and elapsed time.
    """
    start = asyncio.get_event_loop().time()
    try:
        async with session.get(url, timeout=timeout) as resp:
            # read up to a modest amount to simulate extraction without pulling huge files
            await resp.content.read(1024)
            status = resp.status
    except asyncio.TimeoutError:
        return {"url": url, "status": "timeout", "elapsed": None}
    except Exception:
        return {"url": url, "status": "error", "elapsed": None}

    elapsed = asyncio.get_event_loop().time() - start
    return {"url": url, "status": status, "elapsed": elapsed}


async def main(save_baseline: bool = False) -> int:
    """Run a lightweight performance run over URLs in test_data.json.

    Returns 0 on success, 1 on error. Intended for quick runs, not the full
    comprehensive validation.
    """
    base = Path(__file__).parent.parent
    data_file = base / "test_data.json"
    if not data_file.exists():
        print("test_data.json not found; skipping run")
        return 1

    with open(data_file, "r", encoding="utf-8") as f:
        urls = json.load(f)[:NUM_PAPERS]

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        semaphore = asyncio.Semaphore(10)

        async def limited(url):
            async with semaphore:
                return await process_paper(session, url, TIMEOUT_SECONDS)

        tasks = [limited(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total = time.perf_counter() - start
    report = PerformanceReport(total_time=total, avg_time=total / max(1, len(urls)))

    if save_baseline:
        out = base / "performance_baseline.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"total_time": report.total_time}, f)
            print(f"Saved baseline to {out}")
        except Exception:
            print("Failed to save baseline")

    print(f"Completed lightweight performance run: {report.total_time:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
