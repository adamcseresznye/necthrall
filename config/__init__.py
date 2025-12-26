"""Configuration package for Necthrall Lite.

This package re-exports the policy/configuration from :mod:`config.config`
so that `import config` exposes the expected configuration variables.

CRITICAL: Threading optimization (_threading.py) is imported FIRST to ensure
multi-core mode is enabled before numpy/torch/onnxruntime are imported anywhere.
"""

# CRITICAL: Import threading config before anything else
from . import _threading  # noqa: F401 (sets env vars, has side effects)
from .config import Settings, get_settings

__all__ = [
    "get_settings",
    "Settings",
]
