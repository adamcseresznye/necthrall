"""Threading optimization (MUST be imported first).

CRITICAL: This module MUST be imported before any numpy, torch, or onnxruntime
imports occur. On Windows, these libraries lock their thread count at import time
and cannot be changed afterwards.

The config package now imports this at the top of __init__.py, ensuring all
subsequent imports (dotenv, config.py, etc.) happen after threading is configured.
"""

import os
import multiprocessing

# Determine optimal thread count
try:
    num_cores = multiprocessing.cpu_count()
except Exception:
    num_cores = 8

# Force multi-threaded mode for CPU-bound operations
# Must be set BEFORE importing numpy, torch, onnxruntime
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["TORCH_NUM_THREADS"] = str(num_cores)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Log configuration (safe to use print since logging isn't configured yet)
_threading_info = f"⚡ Threading: {num_cores} cores configured (OMP/MKL/TORCH)"
print(_threading_info)
