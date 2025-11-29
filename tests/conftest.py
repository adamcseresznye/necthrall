"""Pytest conftest for Necthrall test configuration.

This file sets up the test environment with proper threading settings
and handles Windows-specific configurations.
"""

import os
import sys

# Set environment variables BEFORE any imports that might use parallel processing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# WINDOWS DLL FIX: Add torch/lib to DLL search path if torch is installed
# This MUST happen before importing torch or any library that uses torch
if os.name == "nt":
    try:
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# CRITICAL: Import torch FIRST before any other library that might load conflicting DLLs
# This prevents DLL conflicts with onnxruntime, sentence-transformers, etc.
# Wrapped in try/except to continue if this fails - tests may still work with mocking
try:
    import torch  # noqa: F401

    _torch_loaded = True
except (ImportError, OSError):
    _torch_loaded = False

# Import fitz (PyMuPDF) early if available - it has its own DLL requirements
try:
    import fitz  # noqa: F401
except ImportError:
    pass
