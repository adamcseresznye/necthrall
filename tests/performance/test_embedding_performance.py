"""Performance tests for ONNX embedding model.

These tests measure embedding performance and are marked as 'performance'
to be skipped by default. Run with: pytest -m performance

Run with: pytest -m performance tests/performance/test_embedding_performance.py
"""

import os
import random
import string
import time
from statistics import mean

import pytest
from loguru import logger

# Lazy import to avoid loading ONNX model at collection time
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Note: Using psutil for memory measurement instead of memory_profiler
# to avoid Windows multiprocessing issues


def _make_scientific_paragraph(min_len=100, max_len=500):
    """Generate a synthetic scientific-like paragraph with token patterns.

    Keeps typical scientific tokens: citations, parentheses, punctuation,
    and technical words to mimic realistic chunk content.
    """
    length = random.randint(min_len, max_len)
    words = []
    while sum(len(w) + 1 for w in words) < length:
        choice = random.random()
        if choice < 0.02:
            words.append("(et al., 2020)")
        elif choice < 0.04:
            words.append("e.g.,")
        elif choice < 0.06:
            words.append("i.e.,")
        elif choice < 0.10:
            words.append("\n")
        else:
            # create a structure with technical tokens
            wlen = random.randint(3, 12)
            word = "".join(random.choices(string.ascii_lowercase, k=wlen))
            if random.random() < 0.05:
                word += "-based"
            words.append(word)
    paragraph = " ".join(words)[:length]
    return paragraph


@pytest.fixture(scope="session")
def chunks_10k():
    """Fixture: generate and return 10,000 realistic text chunks (100-500 chars each).

    Materialize into a list once per session to avoid pytest fixture generator misuse
    and to keep test timing stable.
    """
    random.seed(42)
    n = 10_000
    return [_make_scientific_paragraph(100, 500) for _ in range(n)]


@pytest.fixture(scope="session")
def embedding_model():
    """Instantiate ONNX embedding model once per session.

    Uses ONNX Runtime for 30-60x faster inference vs PyTorch.
    Reuses the same model instance across all tests to avoid repeated loading overhead.
    """
    # Lazy import to avoid DLL conflicts
    from config.onnx_embedding import ONNXEmbeddingModel

    logger.info("Loading ONNX embedding model: {}", MODEL_NAME)
    model = ONNXEmbeddingModel(MODEL_NAME)
    logger.info("ONNX model loaded successfully (expected 30-60x faster than PyTorch)")
    return model


def _measure_embedding_time_and_memory(chunks_list, batch_size, embedding_model):
    """Run embedding using PRODUCTION batched_embed and measure time/memory.

    Returns dict with: total_time, peak_memory_mb, mem_delta_mb
    """
    # Lazy import to avoid DLL conflicts
    from utils.embedding_utils import batched_embed

    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start = time.perf_counter()
    # USE PRODUCTION CODE PATH - tests actual implementation
    embeddings = batched_embed(
        chunks_list,
        embedding_model,
        batch_size=batch_size,
        show_progress=True,  # Enable to see batch timing logs
    )
    total = time.perf_counter() - start

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_delta = mem_after - mem_before

    return {
        "total_time": total,
        "peak_memory_mb": mem_after,
        "mem_delta_mb": mem_delta,
    }


@pytest.mark.performance
def test_10k_chunks_batch32_completes(chunks_10k, embedding_model):
    """Performance test: embed 10k chunks with batch_size=64 on CPU.

    PyTorch baseline: ~490s (20 chunks/sec) - too slow for production
    ONNX target: <20s (500+ chunks/sec) - production ready
    Hard fail: >30s (indicates ONNX migration incomplete)
    Memory: Incremental <300MB (fits Heroku 512MB limit)
    """
    chunks = list(chunks_10k)
    batch_size = 64
    logger.info("Starting embedding run: 10_000 chunks, batch_size={}", batch_size)

    stats = _measure_embedding_time_and_memory(chunks, batch_size, embedding_model)

    total = stats["total_time"]
    peak_mem = stats["peak_memory_mb"]
    mem_delta = stats["mem_delta_mb"]
    throughput = len(chunks) / total if total > 0 else 0

    logger.info(
        "Embedding total time: {total:.3f}s, throughput={throughput:.0f} chunks/sec",
        total=total,
        throughput=throughput,
    )
    logger.info(
        "Memory delta: {delta:.2f} MB (total process: {peak:.2f} MB)",
        delta=mem_delta,
        peak=peak_mem,
    )

    # ONNX target: <20s for 10k chunks (hard fail at 30s)
    # PyTorch baseline: ~490s (acceptable for testing, not production)
    if total >= 30.0:
        logger.error(
            "Performance threshold missed: total={:.1f}s exceeds hard limit of 30s (ONNX migration incomplete?)",
            total,
        )
        pytest.fail(
            f"Embedding took {total:.1f}s which exceeds 30s hard limit. Expected <20s with ONNX Runtime."
        )

    if total < 20.0:
        logger.info(
            "✓ Performance target met: {total:.1f}s < 20s (ONNX optimized)", total=total
        )
    else:
        logger.warning(
            "⚠ Performance below target: {total:.1f}s (expecting <20s with ONNX)",
            total=total,
        )

    # Memory: Incremental usage should be <300MB for Heroku 512MB limit
    assert (
        mem_delta < 300
    ), f"Incremental memory exceeded 300MB: {mem_delta:.1f}MB (will OOM on Heroku)"


@pytest.mark.performance
def test_memory_under_limit(chunks_10k, embedding_model):
    """Verify incremental memory usage <300MB (Heroku 512MB limit compatibility)."""
    chunks = list(chunks_10k)
    stats = _measure_embedding_time_and_memory(
        chunks, batch_size=64, embedding_model=embedding_model
    )
    mem_delta = stats.get("mem_delta_mb")
    peak_mem = stats.get("peak_memory_mb")

    logger.info(
        "Memory usage - Incremental: {delta:.1f}MB, Total: {peak:.1f}MB",
        delta=mem_delta,
        peak=peak_mem,
    )

    assert mem_delta is not None, "No memory profiling result"
    assert (
        mem_delta < 300
    ), f"Incremental memory exceeded 300MB: {mem_delta:.1f}MB (will OOM on Heroku 512MB dyno)"


@pytest.mark.performance
def test_batch_size_comparison(chunks_10k, embedding_model):
    """Compare batch sizes (32, 64, 128) and record timing results."""
    chunks = list(chunks_10k)
    results = {}
    for bs in (32, 64, 128):
        stats = _measure_embedding_time_and_memory(
            chunks, batch_size=bs, embedding_model=embedding_model
        )
        results[bs] = stats["total_time"]
        logger.info("Batch size {} -> total_time {:.3f}s", bs, stats["total_time"])

    # Record fastest batch size
    fastest = min(results, key=results.get)
    logger.info("Fastest batch size: {} (time {:.3f}s)", fastest, results[fastest])
    # No assertion for correctness here; data used in report


@pytest.mark.performance
def test_scaling_behavior(embedding_model):
    """Scaling test across different chunk counts to check approximate linearity.

    Uses 1k, 5k, 10k, 20k chunk counts and measures total time.
    """
    from utils.embedding_utils import batched_embed

    counts = [1_000, 5_000, 10_000, 20_000]
    batch_size = 32
    results = {}
    # Warm up embedding path to avoid measuring one-time costs
    warmup_texts = [_make_scientific_paragraph() for _ in range(128)]
    _ = batched_embed(warmup_texts, embedding_model, batch_size=32)

    for c in counts:
        chunks = [_make_scientific_paragraph() for _ in range(c)]
        stats = _measure_embedding_time_and_memory(
            chunks, batch_size=batch_size, embedding_model=embedding_model
        )
        results[c] = stats["total_time"]
        logger.info("Count {} -> total_time {:.4f}s", c, results[c])

    # Basic check: times should be monotonic increasing with count (no major sub-linear anomaly)
    t1 = results[1_000]
    t5 = results[5_000]
    t10 = results[10_000]
    t20 = results[20_000]
    logger.info(
        "Scaling times: 1k={:.4f}s, 5k={:.4f}s, 10k={:.4f}s, 20k={:.4f}s",
        t1,
        t5,
        t10,
        t20,
    )
    # No strict assertions here; record scaling characteristics for the report.
