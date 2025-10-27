import asyncio
import time
import aiohttp
import json
import fitz  # PyMuPDF
import statistics
import argparse
import uuid
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from memory_profiler import profile

# Configuration
NUM_PAPERS = 30
TEST_ITERATIONS = 5  # Increased for better statistical significance
TIMEOUT_SECONDS = 10
CONCURRENT_LIMIT = 20
TARGET_TIME = 4.0


@dataclass
class PerformanceReport:
    total_time: float
    avg_time_per_paper: float
    success_rate: float
    memory_peak_mb: float
    bottleneck_analysis: Dict[str, float]
    meets_target: bool
    recommendations: List[str]


async def fetch_pdf(session: aiohttp.ClientSession, url: str, timeout: int) -> bytes:
    """Fetches a PDF from a URL."""
    try:
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            return await response.read()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Error fetching {url}: {e}")
        return b""


def extract_text_from_pdf(pdf_data: bytes) -> str:
    """Extracts text from a PDF using PyMuPDF."""
    try:
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except fitz.fitz.PyMuPDFError as e:
        print(f"Error extracting text (likely corrupt PDF): {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during text extraction: {e}")
        return ""


async def process_paper(
    session: aiohttp.ClientSession, url: str, timeout: int
) -> Dict[str, Any]:
    """Fetches a PDF, extracts text, and measures performance."""
    paper_id = str(uuid.uuid4())
    result = {
        "paper_id": paper_id,
        "url": url,
        "success": False,
        "timings": {},
        "error": None,
    }

    try:
        # Download timing
        t_start_download = time.time()
        pdf_data = await fetch_pdf(session, url, timeout)
        t_end_download = time.time()
        result["timings"]["download_start"] = t_start_download
        result["timings"]["download_end"] = t_end_download
        result["timings"]["download_duration"] = t_end_download - t_start_download

        if not pdf_data:
            result["error"] = "Download failed or returned empty content."
            return result

        # Extraction timing
        t_start_extract = time.time()
        extract_text_from_pdf(pdf_data)
        t_end_extract = time.time()
        result["timings"]["extraction_start"] = t_start_extract
        result["timings"]["extraction_end"] = t_end_extract
        result["timings"]["extraction_duration"] = t_end_extract - t_start_extract

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


async def main(
    simulate_timeout: bool = False, concurrency_limit: int = CONCURRENT_LIMIT
) -> PerformanceReport:
    """Main function to run the performance test."""
    timeout = 0.01 if simulate_timeout else TIMEOUT_SECONDS
    if simulate_timeout:
        print("\n--- Running with simulated network timeouts ---")

    with open("test_data.json", "r") as f:
        pdf_urls = json.load(f)

    # Warm-up run
    print("\n--- Running warm-up iteration ---")
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def limited_process_paper_warmup(url):
            async with semaphore:
                return await process_paper(session, url, timeout)

        tasks = [limited_process_paper_warmup(url) for url in pdf_urls[:NUM_PAPERS]]
        await asyncio.gather(*tasks, return_exceptions=True)
    print("--- Warm-up complete ---")

    # Limit the number of papers for the test
    pdf_urls = pdf_urls[:NUM_PAPERS]

    run_id = str(uuid.uuid4())
    log_data = {
        "run_id": run_id,
        "run_timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "num_papers": NUM_PAPERS,
            "test_iterations": TEST_ITERATIONS,
            "concurrency_limit": concurrency_limit,
            "simulate_timeout": simulate_timeout,
        },
        "iterations": [],
    }

    all_results = []
    for i in range(TEST_ITERATIONS):
        print(f"Running iteration {i+1}/{TEST_ITERATIONS}...")
        iteration_results = []
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def limited_process_paper(url):
                async with semaphore:
                    return await process_paper(session, url, timeout)

            tasks = [limited_process_paper(url) for url in pdf_urls]
            iteration_results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time
        all_results.append({"total_time": total_time, "results": iteration_results})
        log_data["iterations"].append(
            {
                "iteration_id": i + 1,
                "total_time": total_time,
                "paper_results": iteration_results,
            }
        )

    # Write detailed log to file
    with open("performance_log.json", "a") as f:
        f.write(json.dumps(log_data, indent=4))
        f.write("\n")
    print("\nDetailed performance log saved to performance_log.json")

    # Analyze results for console report
    total_times = [res["total_time"] for res in all_results]
    avg_total_time = statistics.mean(total_times)
    median_total_time = statistics.median(total_times)
    std_dev_total_time = statistics.stdev(total_times)

    # Filter out exceptions and failed results for analysis
    # Flatten results from all iterations for overall analysis
    all_paper_results = [
        paper_res for iter_res in all_results for paper_res in iter_res["results"]
    ]

    successful_results = [
        res for res in all_paper_results if isinstance(res, dict) and res.get("success")
    ]

    failed_downloads = len(all_results[0]["results"]) - len(successful_results)
    success_rate = len(successful_results) / NUM_PAPERS if NUM_PAPERS > 0 else 0

    if not successful_results:
        print("Warning: No successful downloads to analyze.")
        avg_download_time = 0
        avg_extraction_time = 0
    else:
        avg_download_time = statistics.mean(
            [res["timings"]["download_duration"] for res in successful_results]
        )
        avg_extraction_time = statistics.mean(
            [res["timings"]["extraction_duration"] for res in successful_results]
        )

    report = PerformanceReport(
        total_time=avg_total_time,
        avg_time_per_paper=avg_total_time / NUM_PAPERS,
        success_rate=success_rate,
        memory_peak_mb=0,  # Placeholder
        bottleneck_analysis={
            "download": avg_download_time,
            "extraction": avg_extraction_time,
        },
        meets_target=avg_total_time < TARGET_TIME,
        recommendations=[],  # Placeholder
    )

    print("\n--- Performance Report ---")
    print(
        f"Average total time over {TEST_ITERATIONS} iterations: {report.total_time:.2f} seconds"
    )
    print(f"Median total time: {median_total_time:.2f} seconds")
    print(f"Standard deviation: {std_dev_total_time:.2f} seconds")
    print(f"Average time per paper: {report.avg_time_per_paper:.2f} seconds")
    print(f"Success rate: {report.success_rate:.2%}")
    print(f"Bottleneck Analysis:")
    print(
        f"  - Average download time: {report.bottleneck_analysis['download']:.2f} seconds"
    )
    print(
        f"  - Average extraction time: {report.bottleneck_analysis['extraction']:.2f} seconds"
    )
    print(f"Meets target (< {TARGET_TIME}s): {'Yes' if report.meets_target else 'No'}")

    if not report.meets_target:
        report.recommendations.append(
            "Performance target not met. Consider optimizing the download or extraction process."
        )

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"- {rec}")

    print(
        "\nNote: Memory usage is profiled separately. Check the output from memory_profiler."
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance tests.")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save the results as the new performance baseline.",
    )
    parser.add_argument(
        "--simulate-timeout",
        action="store_true",
        help="Simulate network timeouts.",
    )
    args = parser.parse_args()

    if args.save_baseline:
        baseline_data = {}
        for concurrency in [10, 20, 30]:
            print(f"\n--- Generating baseline for concurrency: {concurrency} ---")
            report = asyncio.run(
                main(
                    simulate_timeout=args.simulate_timeout,
                    concurrency_limit=concurrency,
                )
            )
            baseline_data[f"total_time_concurrency_{concurrency}"] = report.total_time

        # Save the main report's total_time for backward compatibility
        if "total_time" not in baseline_data and "report" in locals():
            baseline_data["total_time"] = report.total_time

        with open("performance_baseline.json", "w") as f:
            json.dump(baseline_data, f, indent=4)
        print("\nPerformance baseline saved.")
    else:
        with open("memory_profile.log", "w") as mem_log:

            @profile(stream=mem_log)
            def run_main_profiled():
                asyncio.run(main(simulate_timeout=args.simulate_timeout))

            run_main_profiled()
        print("Detailed memory usage log saved to memory_profile.log")
