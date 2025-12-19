"""
Configuration management for Necthrall Lite MVP.

Loads environment variables and validates required API keys.
Provides model configurations for LiteLLM routing.
"""

import os
import time
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file (if present).
# During tests we may set SKIP_DOTENV_LOADER=1 to avoid loading a repo .env
# which would interfere with tests that expect missing keys.
if os.getenv("SKIP_DOTENV_LOADER", "").lower() not in ("1", "true", "yes"):
    load_dotenv()

# Configure logger level early
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)

# ============================================================================
# Semantic Scholar API Configuration
# ============================================================================
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# ============================================================================
# LiteLLM Model Configuration - Query Optimization
# Fast models for query transformation and variant generation
# ============================================================================
QUERY_OPTIMIZATION_MODEL = os.getenv("QUERY_OPTIMIZATION_MODEL")
QUERY_OPTIMIZATION_FALLBACK = os.getenv("QUERY_OPTIMIZATION_FALLBACK")

# ============================================================================
# LiteLLM Model Configuration - Answer Synthesis
# Powerful models for final answer generation
# ============================================================================
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL")
SYNTHESIS_FALLBACK = os.getenv("SYNTHESIS_FALLBACK")

# ============================================================================
# API Keys for LiteLLM Providers
# ============================================================================
PRIMARY_LLM_API_KEY = os.getenv("PRIMARY_LLM_API_KEY")
SECONDARY_LLM_API_KEY = os.getenv("SECONDARY_LLM_API_KEY")

# ============================================================================
# RAG Configuration
# ============================================================================
RAG_RETRIEVAL_TOP_K = int(os.getenv("RAG_RETRIEVAL_TOP_K", "50"))
RAG_RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "12"))

# ============================================================================
# Application Settings
# ============================================================================
TIMEOUT = int(os.getenv("TIMEOUT", "30"))  # API timeout in seconds


def _mask_key(value: str | None) -> str:
    """Return a masked representation of an API key for safe debug logging."""
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return value[0:1] + "*" * (len(value) - 1)
    return value[0:4] + "*" * (len(value) - 8) + value[-4:]


def validate_config() -> None:
    """
    Validate that all required environment variables are set.

    Raises:
        ValueError: If any required environment variable is missing

    Note:
        This function is safe to call multiple times; it's cheap and completes
        quickly. It is also called automatically when this module is imported.
    """
    start = time.perf_counter()

    required_vars = {
        "SEMANTIC_SCHOLAR_API_KEY": {
            "value": SEMANTIC_SCHOLAR_API_KEY,
            "help": "Get free key at https://www.semanticscholar.org/product/api",
        },
        "PRIMARY_LLM_API_KEY": {
            "value": PRIMARY_LLM_API_KEY,
            "help": "Get free key at https://www.cerebras.ai/",
        },
        "SECONDARY_LLM_API_KEY": {
            "value": SECONDARY_LLM_API_KEY,
            "help": "Get free key at https://console.groq.com/keys",
        },
    }

    missing = []
    empty_values = []
    for var_name, var_info in required_vars.items():
        val = var_info["value"]
        if val is None:
            missing.append(f"{var_name} - {var_info['help']}")
        elif isinstance(val, str) and val.strip() == "":
            empty_values.append(var_name)

    if empty_values:
        logger.error(
            "Configuration error: environment variables set but empty: {}",
            ", ".join(empty_values),
        )
        raise ValueError(
            "The following environment variables are set but empty: "
            + ", ".join(empty_values)
        )

    if missing:
        error_msg = (
            "Missing required environment variables:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nCreate a .env file in the project root with these variables. See .env.example for a template."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log masked keys and selected configuration for observability
    logger.info("✓ Configuration validated successfully")
    logger.info("  Query optimization model: {}", QUERY_OPTIMIZATION_MODEL)
    logger.info("  Query optimization fallback: {}", QUERY_OPTIMIZATION_FALLBACK)
    logger.info("  Synthesis model: {}", SYNTHESIS_MODEL)
    logger.info("  Synthesis fallback: {}", SYNTHESIS_FALLBACK)
    logger.info("  Timeout: {}s", TIMEOUT)
    logger.debug("  SEMANTIC_SCHOLAR_API_KEY: {}", _mask_key(SEMANTIC_SCHOLAR_API_KEY))
    logger.debug("  PRIMARY_LLM_API_KEY: {}", _mask_key(PRIMARY_LLM_API_KEY))
    logger.debug("  SECONDARY_LLM_API_KEY: {}", _mask_key(SECONDARY_LLM_API_KEY))

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.debug("Configuration validation took %.2fms", elapsed_ms)


# Validate configuration when module is imported
validate_config()
