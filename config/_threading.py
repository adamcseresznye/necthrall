"""Threading optimization.

This module configures threading settings for the application.

The config package now imports this at the top of __init__.py, ensuring all
subsequent imports (dotenv, config.py, etc.) happen after threading is configured.
"""

import multiprocessing
import os

from loguru import logger

# Only set defaults if not already set in environment (e.g. by Docker)
if "OMP_NUM_THREADS" not in os.environ:
    try:
        num_cores = multiprocessing.cpu_count()
    except Exception:
        num_cores = 8

    # In a server context, usually we want this to be 1, but for local scripts
    # we might want max cores.
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["TORCH_NUM_THREADS"] = str(num_cores)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger.info(f"⚡ Threading Config: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
logger.info(f"⚡ Threading Config: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
