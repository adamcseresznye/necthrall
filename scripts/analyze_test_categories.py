#!/usr/bin/env python3
"""
analyze_test_categories.py

Run each test file individually with pytest, measure execution time, infer test counts,
and categorize each file into one of: unit, integration, performance, and a slow marker.

Produces:
 - test_categorization_report.txt (human-readable)
 - test_categorization.json (machine-readable summary)

Usage: python analyze_test_categories.py
"""
import subprocess
import time
import os
import re
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
REPORT_TXT = ROOT / "test_categorization_report.txt"
REPORT_JSON = ROOT / "test_categorization.json"

PYTEST_TIMEOUT = int(os.environ.get("ANALYZE_TEST_TIMEOUT", "300"))  # seconds per file


def find_test_files(tests_dir: Path):
    if not tests_dir.exists():
        return []
    return sorted(
        [
            str(p)
            for p in tests_dir.iterdir()
            if p.is_file() and p.name.startswith("test_") and p.suffix == ".py"
        ]
    )


def extract_test_count(stdout: str, stderr: str):
    """Try several heuristics to extract number of tests run from pytest output."""
    # Try 'collected N items' or 'collected N' patterns
    m = re.search(r"collected\s+(\d+)\s+items", stdout)
    if not m:
        m = re.search(r"collected\s+(\d+)\b", stdout)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # Fall back to summing 'X passed', 'Y failed', 'Z skipped' occurrences
    total = 0
    for s in (stdout + "\n" + stderr).splitlines():
        m = re.search(
            r"^(\s*)(\d+)\s+(passed|failed|skipped|xfailed|xpassed|error)\b",
            s.strip(),
            re.IGNORECASE,
        )
        if m:
            try:
                total += int(m.group(2))
            except Exception:
                pass

    if total > 0:
        return total

    # As a last resort, look for single-test summaries like '1 passed' anywhere
    m = re.findall(r"(\d+)\s+passed", stdout + stderr, flags=re.IGNORECASE)
    if m:
        return sum(int(x) for x in m)

    return 0


def analyze_test_file(test_file: str):
    """Run a single test file and measure execution time and basic stats."""
    start_time = time.perf_counter()
    try:
        # Use python -m pytest to ensure pytest runs even when 'pytest' is not on PATH
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_file,
                "-q",
                "--disable-warnings",
                "--tb=no",
            ],
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT,
        )
        execution_time = time.perf_counter() - start_time
        success = result.returncode == 0
        test_count = extract_test_count(result.stdout, result.stderr)

        return {
            "file": test_file,
            "time": execution_time,
            "success": success,
            "test_count": test_count,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired as ex:
        execution_time = PYTEST_TIMEOUT
        return {
            "file": test_file,
            "time": execution_time,
            "success": False,
            "test_count": 0,
            "stdout": "",
            "stderr": f"TIMEOUT after {PYTEST_TIMEOUT}s",
            "returncode": None,
        }


def categorize_results(results):
    categories = {"unit": [], "integration": [], "performance": [], "slow": []}

    for r in results:
        filename = Path(r["file"]).name
        time_s = r["time"]

        # Performance category (filename hints or long running)
        if "performance" in filename or "benchmark" in filename or time_s > 30:
            categories["performance"].append(r)
        # Integration category (medium length or filename hints)
        elif (
            time_s >= 5
            or "integration" in filename
            or "end_to_end" in filename
            or "end-to-end" in filename
        ):
            categories["integration"].append(r)
        else:
            categories["unit"].append(r)

        # Slow marker (overlaps)
        if time_s >= 10:
            categories["slow"].append(r)

    return categories


def write_reports(categories, results):
    # Human readable text report
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("NECTHRALL TEST CATEGORIZATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        for cat in ("unit", "integration", "performance", "slow"):
            tests = categories.get(cat, [])
            f.write(f"{cat.upper()} TESTS:\n")
            if not tests:
                f.write("  (none)\n")
            else:
                for test in sorted(tests, key=lambda x: x["time"], reverse=True):
                    f.write(
                        f"  - {test['file']}: {test['time']:.1f}s ({test['test_count']} tests)\n"
                    )
            f.write(f"  Total: {len(tests)} files\n\n")

        f.write("SUMMARY:\n")
        total_files = len(results)
        total_tests = sum(r.get("test_count", 0) for r in results)
        total_time = sum(r.get("time", 0) for r in results)
        f.write(f"  Files analyzed: {total_files}\n")
        f.write(f"  Tests found (sum): {total_tests}\n")
        f.write(f"  Total runtime (sum): {total_time:.1f}s\n")

    # JSON report
    out = {
        "summary": {
            "files_analyzed": len(results),
            "total_tests_sum": sum(r.get("test_count", 0) for r in results),
            "total_time_sum_s": sum(r.get("time", 0) for r in results),
        },
        "categories": {
            k: [
                {
                    "file": r["file"],
                    "time": r["time"],
                    "test_count": r["test_count"],
                    "success": r["success"],
                }
                for r in v
            ]
            for k, v in categories.items()
        },
        "results": results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    test_files = find_test_files(TESTS_DIR)
    if not test_files:
        print("No test files found under 'tests/'. Exiting.")
        return

    results = []
    print(
        f"Found {len(test_files)} test files. Starting analysis (timeout per file: {PYTEST_TIMEOUT}s)..."
    )
    for tf in test_files:
        print(f"Analyzing {tf}...")
        r = analyze_test_file(tf)
        results.append(r)
        print(
            f"  → {r['time']:.1f}s, tests: {r['test_count']}, success: {r['success']}"
        )

    categories = categorize_results(results)
    write_reports(categories, results)

    print(
        f"Analysis complete. Reports written to:\n  - {REPORT_TXT}\n  - {REPORT_JSON}"
    )


if __name__ == "__main__":
    main()
