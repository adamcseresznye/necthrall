"""
FINAL SANITY CHECK (CORRECTED)
1. Forces Multi-Threading
2. Applies Windows DLL Fix
3. Handles 'token_type_ids' automatically
"""

import os
import time
import numpy as np

# 1. FORCE MULTI-THREADING
num_cores = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"🚀 STARTING SANITY CHECK with {num_cores} threads...")

# 2. APPLY WINDOWS DLL FIX
try:
    import torch

    print("✓ Torch imported (DLL fix applied)")
except ImportError:
    print("! Torch not found, hoping ONNX works anyway...")

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    print("✓ ONNX Runtime & Tokenizer imported")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: {e}")
    exit(1)


def run_test():
    # Path validation
    model_dir = os.path.join(
        "onnx_model_cache", "sentence-transformers_all-MiniLM-L6-v2"
    )
    model_file = os.path.join(model_dir, "model_quantized.onnx")

    if not os.path.exists(model_file):
        print(f"❌ Model file missing: {model_file}")
        return

    # Load components
    print("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("⏳ Loading ONNX session...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 0
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_file, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # Get expected inputs
    model_input_names = [x.name for x in session.get_inputs()]
    print(f"ℹ️  Model expects inputs: {model_input_names}")

    # Generate Data
    print("⚡ Generating dummy data...")
    texts = [
        "This is a test sentence to benchmark the ONNX runtime speed on CPU " * 4
    ] * 2000

    # Run Inference Loop
    print(f"🔥 Running benchmark on {len(texts)} items (Batch Size 64)...")
    start = time.perf_counter()

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="np"
        )

        # Prepare Input Feed (Dynamic)
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        # FIX: Add token_type_ids if the model asks for them
        if "token_type_ids" in model_input_names:
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            else:
                # Create zeros if tokenizer didn't output them
                ort_inputs["token_type_ids"] = np.zeros_like(
                    inputs["input_ids"], dtype=np.int64
                )

        session.run(None, ort_inputs)

        if i % 320 == 0:
            print(".", end="", flush=True)

    elapsed = time.perf_counter() - start
    throughput = len(texts) / elapsed

    print("\n\n" + "=" * 40)
    print(f"🏁 FINAL RESULT")
    print("=" * 40)
    print(f"⏱️ Time:       {elapsed:.2f}s")
    print(f"🚀 Throughput: {throughput:.0f} chunks/sec")
    print("=" * 40)

    if throughput > 100:
        print("✅ SUCCESS: The issue is your project's config.py file.")
        print(
            "   Action: Remove `os.environ['OMP_NUM_THREADS'] = '1'` from embedding_config.py"
        )
    else:
        print("❌ FAILURE: Hardware limit reached.")


if __name__ == "__main__":
    run_test()
