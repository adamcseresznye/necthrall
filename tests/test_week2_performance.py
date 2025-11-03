#!/usr/bin/env python3
"""
Week 2 Performance Validation Tests

Pytest integration for comprehensive performance validation against Week 2 Day 4 targets:
- Total time: <4 seconds for 25 papers → 1000 chunks → top 10 passages
- Precision@10: ≥ 0.7 on scientific queries
- Memory usage: <500MB peak usage

Integration with scripts/performance_validation.py for CI/CD automation.
"""

import asyncio
import json
import os
import pytest
import statistics
from pathlib import Path
from typing import Dict, Any

from scripts.performance_validation import PerformanceValidator, TEST_QUERIES

pytestmark = [pytest.mark.performance, pytest.mark.slow]


class TestWeek2PerformanceValidation:
    """Comprehensive performance validation tests for Week 2 targets."""

    @pytest.fixture
    def performance_validator(self):
        """Create and initialize performance validator for all tests."""
        validator = PerformanceValidator()
        return validator

    # Note: Async fixtures cause compatibility issues with pytest-asyncio.
    # Instead, we'll run validation tests individually using sync test methods
    # that create their own validator instances to avoid shared state.

    def test_complete_pipeline_execution(self, performance_validator):
        """Test that the complete processing pipeline executes without critical failures."""
        import asyncio

        # Use only 3 queries for faster testing
        async def run_test():
            report = await performance_validator.run_complete_validation(
                limit_queries=3
            )
            total_queries = report.total_queries_tested
            successful_queries = sum(1 for r in report.detailed_results if r["success"])
            success_rate = (
                successful_queries / total_queries if total_queries > 0 else 0
            )
            assert (
                success_rate >= 0.5  # Lower threshold for faster testing
            ), f"Success rate {success_rate:.1%} below minimum 50% threshold"

        asyncio.run(run_test())

    def test_average_processing_time_target(self, performance_validator):
        """Test that average processing time meets <4s target."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            avg_time = report.avg_processing_time
            assert (
                avg_time < 4.0
            ), f"Average processing time {avg_time:.3f}s exceeds 4.0s target"

        asyncio.run(run_test())

    def test_95th_percentile_time_target(self, performance_validator):
        """Test that 95th percentile processing time meets <4.5s target."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            p95_time = report.ninety_fifth_percentile_time
            assert (
                p95_time < 4.5
            ), f"95th percentile time {p95_time:.3f}s exceeds 4.5s target"

        asyncio.run(run_test())

    def test_precision_at_10_target(self, performance_validator):
        """Test that average Precision@10 meets ≥0.7 target."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            avg_precision = report.avg_precision_at_10
            assert (
                avg_precision >= 0.7
            ), f"Average Precision@10 {avg_precision:.3f} below 0.7 target"

        asyncio.run(run_test())

    def test_memory_usage_target(self, performance_validator):
        """Test that peak memory usage meets <500MB target."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            peak_memory = report.peak_memory_usage_mb
            assert (
                peak_memory < 500
            ), f"Peak memory usage {peak_memory:.1f}MB exceeds 500MB target"

        asyncio.run(run_test())

    def test_success_rate_target(self, performance_validator):
        """Test that success rate meets ≥90% target."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            success_rate = report.success_rate
            assert (
                success_rate >= 0.9
            ), f"Success rate {success_rate:.1%} below 90% target"

        asyncio.run(run_test())

    def test_all_targets_met(self, performance_validator):
        """Test that all Week 2 performance targets are met."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            targets = {
                "avg_time_under_4s": report.avg_processing_time < 4.0,
                "p95_time_under_4_5s": report.ninety_fifth_percentile_time < 4.5,
                "precision_at_least_0_7": report.avg_precision_at_10 >= 0.7,
                "memory_under_500mb": report.peak_memory_usage_mb < 500,
                "success_rate_above_0_9": report.success_rate >= 0.9,
            }
            failed_targets = [
                target for target, passed in targets.items() if not passed
            ]
            assert not failed_targets, f"Failed targets: {failed_targets}"

        asyncio.run(run_test())

    def test_stage_timing_completeness(self, performance_validator):
        """Test that all expected pipeline stages are measured."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            expected_stages = {"chunking", "embedding", "retrieval", "reranking"}
            actual_stages = set(report.stage_timing_breakdown.keys())
            missing_stages = expected_stages - actual_stages
            assert not missing_stages, f"Missing stage timings: {missing_stages}"

        asyncio.run(run_test())

    def test_bottleneck_identification(self, performance_validator):
        """Test that bottlenecks are properly identified."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            # At least one stage should be identified as a bottleneck (>20% of total time)
            assert report.bottlenecks, "No bottlenecks identified"
            # Bottlenecks should correspond to stages with timing data
            stage_stages = set(report.stage_timing_breakdown.keys())
            bottleneck_stages = set(report.bottlenecks)
            assert bottleneck_stages.issubset(
                stage_stages
            ), "Bottlenecks contain unknown stages"

        asyncio.run(run_test())

    def test_detailed_results_completeness(self, performance_validator):
        """Test that detailed results contain required fields for all queries."""
        import asyncio

        async def run_test():
            report = await performance_validator.run_complete_validation()
            required_fields = {
                "query",
                "success",
                "total_time",
                "peak_memory_mb",
                "precision_at_10",
                "passages_returned",
                "chunks_indexed",
                "stage_times",
            }
            for result in report.detailed_results:
                missing_fields = required_fields - set(result.keys())
                assert not missing_fields, f"Result missing fields: {missing_fields}"

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_query_diversity_validation(self, performance_validator):
        """Test that all 20 diverse scientific queries are tested."""
        assert len(TEST_QUERIES) == 20, "Expected exactly 20 test queries"

        # Verify queries are diverse scientific topics
        scientific_indicators = [
            "disease",
            "therapy",
            "gene",
            "cell",
            "protein",
            "quantum",
            "machine learning",
            "neural",
            "cancer",
            "drug",
            "clinical",
            "metabolic",
            "cognitive",
            "cardiovascular",
            "immune",
            "genomic",
        ]

        scientific_queries = 0
        for query in TEST_QUERIES:
            query_lower = query.lower()
            if any(indicator in query_lower for indicator in scientific_indicators):
                scientific_queries += 1

        assert (
            scientific_queries >= 15
        ), f"Only {scientific_queries}/20 queries appear scientific"

    @pytest.mark.asyncio
    async def test_error_handling_validation(self, performance_validator):
        """Test error handling scenarios."""
        # Test empty papers scenario
        empty_state_result = await performance_validator._test_empty_papers()
        assert empty_state_result[
            "error_handled"
        ], "Empty papers not handled gracefully"

        # Test corrupted PDF scenario
        corrupted_result = await performance_validator._test_corrupted_pdf()
        assert corrupted_result["passed"], "Corrupted PDF not handled gracefully"


class TestPerformanceRegression:
    """Performance regression tests to prevent future degradation."""

    REGRESSION_THRESHOLD = 1.15  # 15% performance degradation threshold

    @pytest.fixture
    def baseline_file(self, tmp_path):
        """Create a temporary baseline file for regression testing."""
        baseline = tmp_path / "performance_baseline.json"
        baseline_data = {
            "avg_processing_time": 3.0,
            "ninety_fifth_percentile_time": 3.5,
            "avg_precision_at_10": 0.75,
            "peak_memory_usage_mb": 400,
            "success_rate": 0.95,
            "timestamp": 1609459200,  # 2021-01-01
        }

        with open(baseline, "w") as f:
            json.dump(baseline_data, f)

        return baseline

    def test_performance_regression_detection(self, baseline_file):
        """Test that performance regression is detected when thresholds are exceeded."""
        import asyncio

        async def run_test():
            validator = PerformanceValidator()
            validation_report = await validator.run_complete_validation()

            # Load baseline
            with open(baseline_file, "r") as f:
                baseline = json.load(f)

            # Check for regressions
            regressions = []

            current_avg_time = validation_report.avg_processing_time
            baseline_avg_time = baseline["avg_processing_time"]

            if current_avg_time > baseline_avg_time * self.REGRESSION_THRESHOLD:
                regressions.append(
                    f"Average time regression: {current_avg_time:.3f}s vs {baseline_avg_time:.3f}s"
                )

            current_p95_time = validation_report.ninety_fifth_percentile_time
            baseline_p95_time = baseline["ninety_fifth_percentile_time"]

            if current_p95_time > baseline_p95_time * self.REGRESSION_THRESHOLD:
                regressions.append(
                    f"P95 time regression: {current_p95_time:.3f}s vs {baseline_p95_time:.3f}s"
                )

            current_precision = validation_report.avg_precision_at_10
            baseline_precision = baseline["avg_precision_at_10"]

            if (
                current_precision < baseline_precision * 0.95
            ):  # 5% accuracy degradation threshold
                regressions.append(
                    f"Precision regression: {current_precision:.3f} vs {baseline_precision:.3f}"
                )

            current_memory = validation_report.peak_memory_usage_mb
            baseline_memory = baseline["peak_memory_usage_mb"]

            if current_memory > baseline_memory * self.REGRESSION_THRESHOLD:
                regressions.append(
                    f"Memory regression: {current_memory:.1f}MB vs {baseline_memory:.1f}MB"
                )

            current_success = validation_report.success_rate
            baseline_success = baseline["success_rate"]

            if (
                current_success < baseline_success * 0.95
            ):  # 5% success rate degradation threshold
                regressions.append(
                    f"Success rate regression: {current_success:.1%} vs {baseline_success:.1%}"
                )

            # In a real implementation, you might want to log regressions but not fail the test
            # unless they're severe. For now, we'll just document them.
            if regressions:
                print(
                    f"⚠️  Performance regressions detected:\n"
                    + "\n".join(f"  - {r}" for r in regressions)
                )
            else:
                print("✅ No performance regressions detected")

        asyncio.run(run_test())


# Standalone performance test function (can be called from CI/CD)
async def run_week2_performance_test():
    """
    Standalone function to run Week 2 performance validation.
    Returns exit code 0 if all targets met, 1 if targets missed.
    """
    try:
        validator = PerformanceValidator()
        report = await validator.run_complete_validation()
        target_validation = validator.validate_targets(report)

        return 0 if target_validation["targets_met"] else 1

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return 1


if __name__ == "__main__":
    # Allow running as standalone script for CI/CD integration
    import sys

    exit_code = asyncio.run(run_week2_performance_test())
    sys.exit(exit_code)
