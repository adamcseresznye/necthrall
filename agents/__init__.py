"""
Necthrall Lite Agents Package

This package contains all the agents used in the LangGraph workflow:
- SearchAgent: Handles paper search and ranking
- AcquisitionAgent: Handles PDF downloading and text extraction
- FilteringAgent: Handles semantic reranking using embeddings
- ProcessingAgent: Handles embedding generation for content
"""

# Handle PyTorch DLL import issues (PyTorch conflicts with multithreading)
# This needs to run before any agent imports that pull in sentence_transformers/torch
try:
    # Import in this specific order - matches conftest.py working pattern
    import fitz  # PyMuPDF first
    import torch  # PyTorch second

    torch.set_num_threads(1)  # Single-threaded mode to avoid DLL conflicts
    # Import CrossEncoder directly to ensure it's available
    from sentence_transformers import CrossEncoder

    print("✅ Pre-imported PyTorch and related libraries in safe order")
except Exception as e:
    print(f"⚠️  Could not pre-import PyTorch libraries: {e}")
    # Continue - let individual agents handle their own imports

from .search import SearchAgent
from .acquisition import AcquisitionAgent
from .filtering_agent import FilteringAgent
from .processing_agent import ProcessingAgent

__all__ = ["SearchAgent", "AcquisitionAgent", "FilteringAgent", "ProcessingAgent"]
