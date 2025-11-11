"""Configuration package for Necthrall Lite.

This package re-exports the policy/configuration from :mod:`config.config`
so that `import config` exposes the expected configuration variables.
"""

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
