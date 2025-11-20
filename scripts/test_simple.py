"""Test ONNX embedding with exact sanity_check.py approach inside our project structure."""

import os
import time

# FORCE MULTI-THREADING (must be first)
num_cores = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"🚀 TESTING with {num_cores} threads...")

# Windows DLL Fix
try:
    import torch

    print("✓ Torch imported")
except ImportError:
    pass

# Import config to test integration
from config.onnx_embedding import ONNXEmbeddingModel

print("✓ ONNXEmbeddingModel imported")

# Generate test data (same as sanity check)
texts = [
    "This is a test sentence to benchmark the ONNX runtime speed on CPU " * 4
] * 1000

# Test with our model class
print(f"🔥 Running benchmark on {len(texts)} items...")
model = ONNXEmbeddingModel()

start = time.perf_counter()
embeddings = model.get_text_embedding_batch(texts)
elapsed = time.perf_counter() - start

throughput = len(texts) / elapsed

print(f"\n{'='*40}")
print(f"RESULT")
print(f"{'='*40}")
print(f"Time:       {elapsed:.2f}s")
print(f"Throughput: {throughput:.0f} chunks/sec")
print(f"{'='*40}")

if throughput > 200:
    print("✅ SUCCESS: Multi-threading works!")
else:
    print(f"❌ FAIL: Only {throughput:.0f} chunks/sec (expected 300+)")
