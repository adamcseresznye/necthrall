import importlib.util
import os
import numpy as np


def load_hybrid_module():
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "retrieval", "hybrid_retriever.py"
        )
    )
    spec = importlib.util.spec_from_file_location("hybrid_retriever_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_get_faiss_similarities_handles_zero_and_nan():
    """Ensure _get_faiss_similarities normalizes safely for zero/NaN vectors."""
    mod = load_hybrid_module()
    HybridRetriever = mod.HybridRetriever

    hr = HybridRetriever()

    # Attach minimal chunks so k > 0
    hr.chunks = [
        {"content": "a", "embedding": np.zeros(384, dtype=np.float32)} for _ in range(3)
    ]

    # Dummy FAISS index: search returns zeros and valid indices
    class DummyFaiss:
        def __init__(self, ntotal):
            self.ntotal = ntotal

        def search(self, arr, k):
            sims = np.zeros((1, k), dtype=np.float32)
            idx = np.arange(k, dtype=np.int32).reshape(1, k)
            return sims, idx

    hr.faiss_index = DummyFaiss(len(hr.chunks))

    # Zero vector
    q_zero = np.zeros(384, dtype=np.float32)
    sims_zero = hr._get_faiss_similarities(q_zero)
    assert isinstance(sims_zero, np.ndarray)
    assert sims_zero.shape[0] == len(hr.chunks)
    # No NaNs should be present
    assert not np.isnan(sims_zero).any()

    # NaN vector
    q_nan = np.full(384, np.nan, dtype=np.float32)
    sims_nan = hr._get_faiss_similarities(q_nan)
    assert isinstance(sims_nan, np.ndarray)
    assert sims_nan.shape[0] == len(hr.chunks)
    assert not np.isnan(sims_nan).any()

    # Normal random vector should also work
    q_rand = np.random.RandomState(0).randn(384).astype(np.float32)
    sims_rand = hr._get_faiss_similarities(q_rand)
    assert isinstance(sims_rand, np.ndarray)
    assert sims_rand.shape[0] == len(hr.chunks)
    assert not np.isnan(sims_rand).any()
