# tests/conftest.py
import sys
import os
import psutil
import time
import pytest
from typing import Dict, Any, List

# Import fixtures from fixtures.py to make them available
from .fixtures import (
    mock_fastapi_app,
    realistic_scientific_papers,
    realistic_pdf_contents,
    integration_test_queries,
    processing_integration_state,
)

from utils.logging_setup import setup_logging

# Ensure tests use the centralized logging configuration
setup_logging()

# Force CPU-only PyTorch mode
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent deadlocks in concurrent tests


def pytest_configure(config):
    """
    Pre-import modules in safe order to prevent DLL conflicts.
    Executed before pytest collection starts.
    """
    try:
        # Import in this specific order
        import fitz  # PyMuPDF first
        import torch  # PyTorch second

        torch.set_num_threads(1)  # Single-threaded mode
        from sentence_transformers import SentenceTransformer  # Last

        print("SUCCESS: Pre-imported problematic modules successfully")
    except Exception as e:
        print(f"WARNING: Could not pre-import modules: {e}")
        # Don't fail - let tests attempt to run

    # Register custom marks
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "pdf_dependent: marks tests as requiring PDF processing"
    )
    config.addinivalue_line("markers", "timeout(seconds): marks tests with a timeout")
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring full pipeline",
    )
    config.addinivalue_line(
        "markers", "performance: marks tests focused on performance benchmarking"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks fast unit tests or can be used with '-m unit' to select fast tests",
    )


def pytest_collection_modifyitems(config, items):
    """
    Custom selection behavior: treat `-m unit` as "unit OR not (integration or performance or slow)".
    This lets users run `pytest -m "unit"` and get fast tests even if many tests are unmarked.
    """
    mexpr = config.getoption("-m")
    if not mexpr:
        return

    # Quick textual check for the unit request; keep simple behaviour for common cases
    if "unit" in mexpr and "not unit" not in mexpr:
        selected = []
        deselected = []
        for item in items:
            # If explicitly marked unit, include
            if item.get_closest_marker("unit"):
                selected.append(item)
                continue

            # If explicitly marked as performance/slow, deselect unless also unit
            if item.get_closest_marker("performance") or item.get_closest_marker(
                "slow"
            ):
                # keep if also explicitly unit
                if item.get_closest_marker("unit"):
                    selected.append(item)
                else:
                    deselected.append(item)
            else:
                # Unmarked tests are treated as unit tests
                selected.append(item)

        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(scope="session")
def performance_monitor():
    """Global performance monitor for test suite."""
    return PerformanceMonitor()


class PerformanceMonitor:
    """Monitor performance across the test suite."""

    def __init__(self):
        self.start_time = time.time()
        self.test_results: List[Dict[str, Any]] = []
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)

    def record_test_result(self, test_name: str, result: Dict[str, Any]):
        """Record a test result for final reporting."""
        self.test_results.append(
            {
                "name": test_name,
                "passed": result.get("passed", False),
                "time": result.get("time", 0.0),
                "memory_delta": result.get("memory_delta", 0.0),
                "details": result.get("details", {}),
                "timestamp": time.time(),
            }
        )

    def get_suite_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire test suite."""
        if not self.test_results:
            return {"error": "No test results recorded"}

        total_time = time.time() - self.start_time
        passed_tests = sum(1 for r in self.test_results if r["passed"])
        total_tests = len(self.test_results)

        # Performance metrics
        avg_test_time = sum(r["time"] for r in self.test_results) / total_tests
        max_memory_delta = max(r["memory_delta"] for r in self.test_results)
        total_memory_delta = (
            sum(r["memory_delta"] for r in self.test_results) / total_tests
        )

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "suite_time": total_time,
            "avg_test_time": avg_test_time,
            "max_memory_delta": max_memory_delta,
            "avg_memory_delta": total_memory_delta,
            "passed": passed_tests == total_tests,
        }

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        stats = self.get_suite_stats()

        report_lines = [
            "=" * 80,
            "INTEGRATION TEST SUITE PERFORMANCE REPORT",
            "=" * 80,
            "",
            f"Total Tests: {stats['total_tests']}",
            f"Passed: {stats['passed_tests']}, Failed: {stats['failed_tests']}",
            f"Suite Time: {stats['suite_time']:.3f}s",
            f"Average Test Time: {stats['avg_test_time']:.3f}s",
            f"Peak Memory Usage: +{stats['max_memory_delta']:.1f}MB",
            f"Average Memory Delta: +{stats['avg_memory_delta']:.1f}MB",
            "",
        ]

        # Performance target validation
        targets = {
            "Suite Time < 30s": stats["suite_time"] < 30.0,
            "Average Test Time < 5s": stats["avg_test_time"] < 5.0,
            "Memory Usage < 1000MB": stats["max_memory_delta"] < 1000.0,
            "All Tests Pass": stats["passed"],
        }

        report_lines.append("PERFORMANCE TARGET VALIDATION:")
        for target, met in targets.items():
            status = "✅ PASS" if met else "❌ FAIL"
            report_lines.append(f"  {status}: {target}")

        report_lines.append("")

        # Detailed test results
        report_lines.extend(
            [
                "=" * 80,
                "DETAILED TEST RESULTS",
                "=" * 80,
                "",
            ]
        )

        for i, result in enumerate(self.test_results, 1):
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            report_lines.append(
                f"{i:2d}. {status} {result['name'][:60]} "
                f"({result['time']:.3f}s, +{result['memory_delta']:.1f}MB)"
            )

            # Add failure details if present
            if not result["passed"] and result["details"]:
                for key, value in result["details"].items():
                    report_lines.append(f"      {key}: {value}")
                report_lines.append("")

        return "\n".join(report_lines)


@pytest.fixture(scope="function")
def test_timer(performance_monitor):
    """Timer fixture for individual test performance tracking."""
    return TestTimer(performance_monitor)


class TestTimer:
    """Timer for individual test performance measurement."""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.start_time = None
        self.initial_memory = None
        self.process = psutil.Process()

    def start(self):
        """Start timing the test."""
        self.start_time = time.perf_counter()
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)

    def stop(
        self, test_name: str = None, passed: bool = True, details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Stop timing and record results."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        end_time = time.perf_counter()
        final_memory = self.process.memory_info().rss / (1024 * 1024)

        result = {
            "passed": passed,
            "time": end_time - self.start_time,
            "memory_delta": final_memory - self.initial_memory,
            "details": details or {},
        }

        if self.monitor and test_name:
            self.monitor.record_test_result(test_name, result)

        return result


@pytest.fixture(scope="session", autouse=True)
def session_report(performance_monitor, request):
    """Generate final performance report at session end."""

    def generate_final_report():
        if performance_monitor.test_results:
            report = performance_monitor.generate_report()
            stats = performance_monitor.get_suite_stats()

            # Save to file
            report_file = "test_performance_report.txt"
            try:
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(report)
                    f.write(f"\n\nGenerated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"\n📊 Performance report saved to: {report_file}")
            except Exception as e:
                print(f"\n⚠️  Failed to save performance report: {e}")

            # Console summary
            print(f"\n{'='*60}")
            print("TEST SUITE SUMMARY")
            print(f"{'='*60}")
            status = (
                "✅ ALL TESTS PASSED" if stats["passed"] else "❌ SOME TESTS FAILED"
            )
            print(f"Status: {status}")
            print(f"Tests: {stats['passed_tests']}/{stats['total_tests']} passed")
            print(
                f"Time: {stats['suite_time']:.3f}s total, {stats['avg_test_time']:.3f}s average"
            )
            print(
                f"Memory: +{stats['avg_memory_delta']:.1f}MB average, +{stats['max_memory_delta']:.1f}MB peak"
            )
            print(f"{'='*60}")

    request.addfinalizer(generate_final_report)
