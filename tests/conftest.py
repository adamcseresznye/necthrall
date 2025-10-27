# tests/conftest.py
import sys
import os

# Force CPU-only PyTorch mode
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def pytest_configure(config):
    """
    Pre-import modules in safe order to prevent DLL conflicts.
    Executed before pytest collection starts.
    """
    try:
        # Import in this specific order
        import fitz  # PyMuPDF first
        import torch  # PyTorch second

        torch.set_num_threads(1)  # Single-threaded mode
        from sentence_transformers import SentenceTransformer  # Last

        print("SUCCESS: Pre-imported problematic modules successfully")
    except Exception as e:
        print(f"WARNING: Could not pre-import modules: {e}")
        # Don't fail - let tests attempt to run
