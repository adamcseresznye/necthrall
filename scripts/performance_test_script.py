#!/usr/bin/env python3
"""
Performance test for optimized CrossEncoderReranker.
Tests the performance target of <600ms for 15 passage pairs.
"""

import time
from retrieval.reranker import CrossEncoderReranker


def test_reranker_performance():
    """Test CrossEncoderReranker meets 600ms performance target."""

    # Create reranker (eager loading + warmup happens here)
    start_init = time.perf_counter()
    reranker = CrossEncoderReranker()
    init_time = (time.perf_counter() - start_init) * 1000

    print(".2f")

    # Create test passages (15 as target)
    passages = [
        {
            "content": f"This is test passage {i} with some content about machine learning and computer vision tasks. "
            * 5,
            "doc_id": i,
            "retrieval_score": 0.9
            - (i * 0.05),  # Decreasing scores to ensure reranking happens
        }
        for i in range(15)
    ]

    query = "machine learning and computer vision applications"

    print("\nTesting reranking of 15 passages...")
    # Measure reranking time
    start_rerank = time.perf_counter()
    results, metrics = reranker.rerank(query, passages, return_metrics=True)
    rerank_time = (time.perf_counter() - start_rerank) * 1000

    print(f"Reranking time: {rerank_time:.2f}ms")
    print(f"Performance target met: {rerank_time < 600}")
    print(f"Skip rate: {metrics.get('skip_rate', 0):.2%}")
    print(f"Number of results returned: {len(results)}")

    # Test multiple reranks to see skip behavior
    print("\nPerformance Summary:")
    print(f"Initialization time: {init_time:.2f}ms (target: <3000ms reset threshold)")
    print(f"Single rerank: {rerank_time:.2f}ms (target: <600ms)")
    print(f"Status: {'✅ PASSED' if rerank_time < 600 else '❌ FAILED'}")

    return rerank_time < 600


if __name__ == "__main__":
    success = test_reranker_performance()
    exit(0 if success else 1)
