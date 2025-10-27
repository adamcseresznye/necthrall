import asyncio
import json
import statistics
import pytest
from pathlib import Path
from scripts.performance_test import main as run_performance_test, PerformanceReport

BASELINE_FILE = Path(__file__).parent.parent / "performance_baseline.json"
REGRESSION_THRESHOLD = 1.20  # 20% slowdown


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


async def run_performance_test_with_params(concurrency: int):
    """Helper to run performance test with a given concurrency."""
    from scripts.performance_test import (
        NUM_PAPERS,
        TEST_ITERATIONS,
        TIMEOUT_SECONDS,
        process_paper,
    )
    import aiohttp

    with open("test_data.json", "r") as f:
        pdf_urls = json.load(f)[:NUM_PAPERS]

    all_times = []
    for _ in range(TEST_ITERATIONS):
        start_time = asyncio.get_event_loop().time()
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)

            async def limited_process_paper(url):
                async with semaphore:
                    return await process_paper(session, url, TIMEOUT_SECONDS)

            tasks = [limited_process_paper(url) for url in pdf_urls]
            await asyncio.gather(*tasks, return_exceptions=True)

        total_time = asyncio.get_event_loop().time() - start_time
        all_times.append(total_time)

    return {"total_time": statistics.mean(all_times)}


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [10, 20, 30])
async def test_performance_regression(concurrency):
    """Tests for performance regression against a baseline."""
    if not BASELINE_FILE.exists():
        pytest.skip(
            "Baseline file not found. Run 'python scripts/performance_test.py --save-baseline' to create it."
        )

    with open(BASELINE_FILE, "r") as f:
        baseline = json.load(f)

    results = await run_performance_test_with_params(concurrency)
    current_time = results["total_time"]
    baseline_time = baseline.get(
        f"total_time_concurrency_{concurrency}", baseline["total_time"]
    )

    print(f"Current average time: {current_time:.2f}s")
    print(f"Baseline average time: {baseline_time:.2f}s")

    assert (
        current_time < baseline_time * REGRESSION_THRESHOLD
    ), f"Performance regression detected! Time increased by more than 20%."


def test_placeholder_for_sync_run():
    # Pytest needs at least one synchronous test to avoid warnings if all are async
    pass
