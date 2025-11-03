import importlib.util
import os
import sys
import numpy as np

# Load module directly from file to avoid package-level imports that pull heavy deps
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "retrieval", "hybrid_retriever.py")
)
spec = importlib.util.spec_from_file_location("hybrid_retriever_mod", module_path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print("Failed to load module:", e)
    raise

HybridRetriever = mod.HybridRetriever

# Instantiate
hr = HybridRetriever()

# Create 3 dummy chunks with embeddings
hr.chunks = [
    {"content": f"doc{i}", "embedding": np.zeros(384, dtype=np.float32)}
    for i in range(3)
]


# Dummy FAISS index with a search method
class DummyFaiss:
    def __init__(self, ntotal):
        self.ntotal = ntotal

    def search(self, arr, k):
        # return similarities (1,k) and indices (1,k)
        sims = np.zeros((1, k), dtype=np.float32)
        idx = np.arange(k, dtype=np.int32).reshape(1, k)
        return sims, idx


hr.faiss_index = DummyFaiss(len(hr.chunks))

print("---- Test with zero vector ----")
q = np.zeros(384, dtype=np.float32)
try:
    sims = hr._get_faiss_similarities(q)
    print("sims shape:", sims.shape, "sims sample:", sims[:5])
except Exception as e:
    print("Error for zero vector:", e)

print("\n---- Test with NaN vector ----")
q_nan = np.full(384, np.nan, dtype=np.float32)
try:
    sims2 = hr._get_faiss_similarities(q_nan)
    print("sims2 shape:", sims2.shape, "any NaN in sims2:", np.isnan(sims2).any())
except Exception as e:
    print("Error for NaN vector:", e)

print("\n---- Test with normal small random vector ----")
q3 = np.random.RandomState(0).randn(384).astype(np.float32)
try:
    sims3 = hr._get_faiss_similarities(q3)
    print("sims3 shape:", sims3.shape, "sims3 sample:", sims3[:5])
except Exception as e:
    print("Error for random vector:", e)

print("\nDone")
