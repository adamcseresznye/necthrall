"""Configuration package for Necthrall Lite.

This package re-exports the policy/configuration from :mod:`config.config`
so that `import config` exposes the expected configuration variables.

CRITICAL: Threading optimization (_threading.py) is imported FIRST to ensure
multi-core mode is enabled before numpy/torch/onnxruntime are imported anywhere.
"""

# CRITICAL: Import threading config before anything else
from . import _threading  # noqa: F401 (sets env vars, has side effects)

from .config import *  # noqa: F401,F403  (re-export configuration symbols)

__all__ = [
    # re-exported names (not exhaustive) - provide common entry points
    "SEMANTIC_SCHOLAR_API_KEY",
    "QUERY_OPTIMIZATION_MODEL",
    "QUERY_OPTIMIZATION_FALLBACK",
    "SYNTHESIS_MODEL",
    "SYNTHESIS_FALLBACK",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "TIMEOUT",
    "validate_config",
]
