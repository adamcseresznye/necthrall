"""Pytest conftest to configure Windows DLL search path before imports.

This attempts to add the `torch/lib` directory (inside the active venv)
to the DLL search path using `os.add_dll_directory` on Windows. It runs at
collection time, so it executes before other test modules import packages
that may trigger `torch`/`transformers` imports and avoids DLL init errors.
"""

import os
import sys
import types

# Provide a lightweight stub for `transformers` to avoid importing the
# real package (which imports `torch` at import-time and can trigger
# Windows DLL initialization failures during test collection).
if "transformers" not in sys.modules:
    mod = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            # Return a callable tokenizer that mirrors the subset of the
            # interface used by `ONNXEmbeddingModel`: __call__ returning
            # `input_ids` and `attention_mask` numpy arrays.
            def tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            ):
                import numpy as _np

                n = len(texts)
                # simple shapes: (n, 1) so downstream code still works
                input_ids = _np.zeros((n, 1), dtype=_np.int64)
                attention_mask = _np.ones((n, 1), dtype=_np.int64)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            return tokenizer

    mod.AutoTokenizer = AutoTokenizer
    sys.modules["transformers"] = mod

# Provide a lightweight stub for `torch` to avoid loading real torch DLLs
# during test collection on Windows. Some modules import `torch` at import
# time which triggers DLL initialization; substituting a harmless module
# prevents that. Tests that need real torch should opt-in separately.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")


if os.name == "nt":
    try:
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.isdir(torch_lib):
            try:
                # Preferred on Python 3.8+: adds directory to DLL search path
                os.add_dll_directory(torch_lib)
            except Exception:
                # Fallback: prepend to PATH so Windows can locate dependencies
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # Keep tests resilient; if this fails we'll let imports surface errors
        pass
"""
Pytest configuration and fixtures.

CRITICAL: Import order to avoid torch DLL initialization errors on Windows.
This must be the first import in any test session.
See: https://github.com/pytorch/pytorch/issues/91966
"""

# Set environment variables BEFORE any imports that might use torch
import os


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import torch FIRST before any other ML libraries
# Commented out for ONNX Runtime testing - torch DLL issues on Windows
# try:
#     import torch
# except ImportError:
#     pass  # torch may not be installed in all test environments

# Now safe to import fitz, sentence-transformers, etc.
try:
    import fitz  # PyMuPDF
except ImportError:
    pass

# Commented out for ONNX Runtime - avoiding torch DLL dependency
# try:
#     import sentence_transformers
# except ImportError:
#     pass
