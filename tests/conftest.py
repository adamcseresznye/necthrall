"""
Pytest configuration and fixtures.

CRITICAL: Import order to avoid torch DLL initialization errors on Windows.
This must be the first import in any test session.
See: https://github.com/pytorch/pytorch/issues/91966
"""

# Set environment variables BEFORE any imports that might use torch
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import torch FIRST before any other ML libraries
try:
    import torch
except ImportError:
    pass  # torch may not be installed in all test environments

# Now safe to import fitz, sentence-transformers, etc.
try:
    import fitz  # PyMuPDF
except ImportError:
    pass

try:
    import sentence_transformers
except ImportError:
    pass
