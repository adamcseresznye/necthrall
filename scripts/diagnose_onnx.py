"""Diagnose ONNX Runtime threading and performance.

Checks:
1. Environment variables
2. ONNX Runtime configuration
3. Actual CPU utilization during inference
"""

import os
import psutil
import time
from config.onnx_embedding import ONNXEmbeddingModel

print("=" * 80)
print("ONNX Runtime Diagnostic")
print("=" * 80)

# Check environment
print("\n1. Environment Variables:")
print(f"   OMP_NUM_THREADS:  {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"   MKL_NUM_THREADS:  {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
print(f"   TORCH_NUM_THREADS: {os.environ.get('TORCH_NUM_THREADS', 'NOT SET')}")

# Load model
print("\n2. Loading ONNX Model...")
model = ONNXEmbeddingModel()

# Check session config
print(f"\n3. ONNX Session Configuration:")
print(
    f"   intra_op_num_threads: {model.session.get_session_options().intra_op_num_threads}"
)
print(
    f"   inter_op_num_threads: {model.session.get_session_options().inter_op_num_threads}"
)
print(f"   Execution providers: {model.session.get_providers()}")

# CPU usage test
print(f"\n4. CPU Utilization Test:")
print("   Running 100-chunk batch...")

test_texts = ["This is a test sentence for embedding."] * 100

# Measure CPU before
cpu_before = psutil.cpu_percent(interval=0.5, percpu=True)
print(f"   CPU before: {sum(cpu_before)/len(cpu_before):.1f}% avg")

# Run inference
start = time.time()
embeddings = model.get_text_embedding_batch(test_texts)
elapsed = time.time() - start

# Measure CPU after
cpu_after = psutil.cpu_percent(interval=0.5, percpu=True)
print(f"   CPU during: {sum(cpu_after)/len(cpu_after):.1f}% avg")
print(f"   Per-core usage: {[f'{c:.0f}%' for c in cpu_after]}")

throughput = len(test_texts) / elapsed
print(f"\n5. Results:")
print(f"   Time: {elapsed:.3f}s")
print(f"   Throughput: {throughput:.1f} chunks/sec")

# Diagnosis
print(f"\n6. Diagnosis:")
if sum(cpu_after) / len(cpu_after) < 30:
    print("   ❌ Low CPU usage - single-threaded or I/O bound")
elif max(cpu_after) > 80 and sum(cpu_after) / len(cpu_after) < 40:
    print("   ❌ One core at 100%, others idle - single-threaded execution")
elif sum(cpu_after) / len(cpu_after) > 60:
    print("   ✅ Multi-core utilization detected!")
else:
    print("   ⚠️  Moderate usage - partial multi-threading")

print("\n" + "=" * 80)
