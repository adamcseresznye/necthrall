#!/usr/bin/env python3
"""
Quick performance test for single query validation
"""

import asyncio
import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.performance_validation import PerformanceValidator


async def quick_test():
    """Run quick test with just one query."""
    print("🚀 Running quick performance test with 1 query...")

    validator = PerformanceValidator()

    # Override to just test 1 query
    validator.test_queries = ["cardiovascular risks of intermittent fasting"]

    # Run validation
    print("📊 Testing query: cardiovascular risks of intermittent fasting")
    result = await validator._run_query_validation(
        "cardiovascular risks of intermittent fasting"
    )

    print("✅ Result:")
    print(f"   Success: {result.success}")
    print(f"   Total time: {result.total_time:.3f}s")
    print(f"   Memory: {result.peak_memory_mb:.1f}MB")
    print(f"   Precision@10: {result.precision_at_10}")
    print(f"   Passages: {result.passages_returned}")
    print(f"   Chunks: {result.chunks_indexed}")

    if result.error:
        print(f"   ❌ Error: {result.error}")

    print("\n⏱️ Stage times:")
    for stage, time_val in result.stage_times.items():
        print(f"   • {stage}: {time_val:.3f}s")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
