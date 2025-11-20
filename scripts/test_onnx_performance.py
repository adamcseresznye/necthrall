"""Standalone ONNX performance test - Multi-Core with Batching.

CRITICAL: Environment variables MUST be set before ANY imports (matching sanity_check.py pattern).
"""

import os
import time

# CRITICAL: Set threading BEFORE any library imports (including config)
num_cores = os.cpu_count() or 16
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["TORCH_NUM_THREADS"] = str(num_cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"⚡ Pre-configured threading: {num_cores} cores")

# NOW safe to import everything
import psutil
import random
import string
import numpy as np

# Import ONNX model
try:
    from config.onnx_embedding import ONNXEmbeddingModel

    print("✓ Successfully imported ONNXEmbeddingModel")
except Exception as e:
    print(f"✗ Failed to import ONNX model: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("=" * 80)
print("ONNX Runtime Performance Test (Multi-Core + Batching)")
print("=" * 80)

# Initialize model
print("\nInitializing ONNX model...")
start_init = time.perf_counter()
model = ONNXEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
init_time = time.perf_counter() - start_init
print(f"✓ Model initialized in {init_time:.2f}s")

# Generate test data (matching sanity_check.py pattern for fair comparison)
TEST_SIZE = 1000
print(f"\nGenerating {TEST_SIZE} test chunks...")

# Use same pattern as sanity_check: repeated sentence for consistent performance
test_sentence = (
    "This is a test sentence to benchmark the ONNX runtime speed on CPU " * 4
)
chunks = [test_sentence] * TEST_SIZE

print(f"✓ Generated {len(chunks)} chunks")

# Measure performance
print("\nRunning embedding performance test...")
print(f"Target: ~300+ chunks/sec (< 6s for {TEST_SIZE} chunks with multi-threading)")

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

# --- BATCHING LOGIC ---
BATCH_SIZE = 64
total_chunks = len(chunks)
all_embeddings = []

start = time.perf_counter()
try:
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_embs = model.get_text_embedding_batch(batch)
        all_embeddings.extend(batch_embs)

        if i % 1000 == 0:
            print(".", end="", flush=True)

    elapsed = time.perf_counter() - start
    print("\n")

    mem_after = process.memory_info().rss / 1024 / 1024
    mem_delta = mem_after - mem_before
    throughput = total_chunks / elapsed

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total time:       {elapsed:.2f}s")
    print(f"Throughput:       {throughput:.0f} chunks/sec")
    print(f"Memory delta:     {mem_delta:.1f} MB")
    print(f"Total memory:     {mem_after:.1f} MB")

    # Extrapolate to 10k chunks
    time_for_10k = elapsed * (10000 / TEST_SIZE)
    print(
        f"\nExtrapolated 10k: {time_for_10k:.1f}s ({10000/time_for_10k:.0f} chunks/sec)"
    )

    # Performance tiers
    if throughput > 300:
        print(f"\n✅ EXCELLENT: Matching sanity check! ({throughput:.0f} chunks/sec)")
    elif throughput > 150:
        print(
            f"\n✅ PASS: Multi-core working ({throughput:.0f} chunks/sec, 7x faster than baseline)"
        )
    elif throughput > 80:
        print(
            f"\n⚠️  PARTIAL: Some speedup ({throughput:.0f} chunks/sec, but below 150 target)"
        )
    else:
        print(
            f"\n❌ FAIL: Single-core mode ({throughput:.0f} chunks/sec, baseline ~20)"
        )

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
