#!/usr/bin/env python3
"""
Batch Size Optimization Benchmark for CrossEncoderReranker.

Tests different batch sizes (5, 10, 15, 20) to find optimal performance trade-offs
for latency vs memory usage in cross-encoder reranking.
"""
# Force early imports to avoid issues later
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Import in this specific order to handle PyTorch DLL issues
    import torch  # PyTorch first

    torch.set_num_threads(1)  # Single-threaded mode to avoid conflicts
    from sentence_transformers import (
        CrossEncoder,
    )  # Import CrossEncoder directly to avoid circular imports

    print("SUCCESS: Pre-imported problematic modules successfully")
except Exception as e:
    print(f"WARNING: Could not pre-import modules: {e}")
    # Don't fail - let benchmark attempt to run

import time
import json
from typing import Dict, List, Any
from retrieval.reranker import CrossEncoderReranker


def benchmark_batch_sizes() -> Dict[str, Any]:
    """Benchmark different batch sizes for cross-encoder reranking."""

    results = {
        "batch_sizes_tested": [5, 10, 15, 20],
        "results": {},
        "recommendations": {},
    }

    # Test passages (similar content to avoid bias)
    test_passages = [
        {
            "content": f"This is test passage {i} about machine learning algorithms and neural networks in scientific research applications. "
            * 10,
            "doc_id": i,
            "retrieval_score": 0.8 - (i * 0.02),  # Gradual score decrease
        }
        for i in range(20)  # Enough for all batch sizes
    ]

    query = "machine learning algorithms neural networks"

    for batch_size in results["batch_sizes_tested"]:
        print(f"\n🏃 Testing batch size: {batch_size}")

        # Configure reranker for this batch size
        try:
            # Temporarily modify batch size for testing (since model is already loaded)
            reranker = CrossEncoderReranker()
            original_batch_size = reranker.batch_size
            reranker.batch_size = batch_size

            # Test multiple runs for stability
            times = []
            memories = []
            successful_runs = 0

            for run in range(5):  # 5 runs for stability
                try:
                    start_time = time.perf_counter()

                    # Rerank with metrics
                    results_list, metrics = reranker.rerank(
                        query,
                        test_passages[
                            : min(20, batch_size * 3)
                        ],  # Ensure enough for meaningful batching
                        return_metrics=True,
                    )

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    times.append(elapsed_ms)
                    memories.append(metrics.get("memory_peak_mb", 0))
                    successful_runs += 1

                    print(f"  🔄 Run {run + 1}: {elapsed_ms:.1f}ms")

                except Exception as e:
                    print(f"  ⚠️  Run {run + 1} failed: {e}")
                    continue

            if successful_runs > 0:
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories) if memories else 0
                success_rate = successful_runs / 5

                results["results"][str(batch_size)] = {
                    "batch_size": batch_size,
                    "avg_time_ms": round(avg_time, 2),
                    "avg_memory_mb": round(avg_memory, 2),
                    "success_rate": round(success_rate, 3),
                    "successful_runs": successful_runs,
                    "meets_target": avg_time < 600,  # 600ms target
                }

                print(
                    f"  ✅ Average: {avg_time:.1f}ms, {avg_memory:.1f}MB, {success_rate:.1%} success"
                )
            else:
                results["results"][str(batch_size)] = {
                    "batch_size": batch_size,
                    "error": "All runs failed",
                }
                print(f"  ❌ All runs failed for batch size {batch_size}")

        except Exception as e:
            results["results"][str(batch_size)] = {
                "batch_size": batch_size,
                "error": str(e),
            }
            print(f"  ❌ Batch size {batch_size} setup failed: {e}")

    # Generate recommendations
    valid_results = {
        bs: data
        for bs, data in results["results"].items()
        if "avg_time_ms" in data and data["meets_target"]
    }

    if valid_results:
        # Find best batch size (balance of speed and memory)
        best_batch_size = min(
            valid_results.keys(),
            key=lambda bs: valid_results[bs]["avg_time_ms"]
            + valid_results[bs]["avg_memory_mb"] * 10,
        )  # Weight memory more

        results["recommendations"] = {
            "recommended_batch_size": int(best_batch_size),
            "performance_profile": valid_results[best_batch_size],
            "basis": "optimal balance of latency and memory usage among sizes meeting 600ms target",
            "available_options": list(valid_results.keys()),
        }

        print(
            f"\n🎯 Recommendation: Batch size {best_batch_size} "
            f"({valid_results[best_batch_size]['avg_time_ms']:.1f}ms, "
            f"{valid_results[best_batch_size]['avg_memory_mb']:.1f}MB)"
        )
    else:
        results["recommendations"] = {
            "error": "No batch size met the 600ms performance target",
            "suggestion": "Consider model optimization or hardware improvements",
        }

    return results


def main():
    """Run batch size benchmark and save results."""
    print("🔬 CrossEncoderReranker Batch Size Optimization Benchmark")
    print("=" * 60)

    try:
        results = benchmark_batch_sizes()

        # Save results
        with open("batch_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results saved to batch_benchmark_results.json")

        # Print summary
        print("\n📈 Summary:")
        for bs, data in results["results"].items():
            if "avg_time_ms" in data:
                status = "✅" if data["meets_target"] else "❌"
                print(f"  Batch {bs}: {data['avg_time_ms']:.1f}ms {status}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


if __name__ == "__main__":
    main()
