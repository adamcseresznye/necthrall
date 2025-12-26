"""Embedding configuration and startup helpers.

Initializes the high-performance ONNX embedding model and stores it
on `app.state.embedding_model` for reuse across requests.

Usage:
    from fastapi import FastAPI
    from config.embedding_config import init_embedding, get_embedding_model

    app = FastAPI()

    @app.on_event("startup")
    async def startup():
        await init_embedding(app)

Error Handling:
    - ImportError (onnxruntime missing): Logs critical warning, app continues without embeddings.
    - RuntimeError (model file missing): Logs error with setup instructions, app continues.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI
from loguru import logger

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384

# Module-level flags for lazy import state
# These are set during first call to init_embedding
_onnx_available: Optional[bool] = None
_init_onnx: Optional[callable] = None


def _lazy_import_onnx() -> tuple[bool, Optional[callable]]:
    """Lazily import the ONNX embedding module.

    Returns:
        Tuple of (is_available, init_function)
    """
    global _onnx_available, _init_onnx

    if _onnx_available is not None:
        return _onnx_available, _init_onnx

    try:
        from config.onnx_embedding import initialize_embedding_model

        _onnx_available = True
        _init_onnx = initialize_embedding_model
        return True, initialize_embedding_model
    except ImportError as e:
        _onnx_available = False
        _init_onnx = None
        logger.critical(
            f"onnxruntime is not installed. Embedding features will be disabled. "
            f"Install with: pip install onnxruntime. Error: {e}"
        )
        return False, None


async def init_embedding() -> Optional[Any]:
    """Initialize ONNX Embedding Model.

    Returns:
        The initialized embedding model or None if initialization failed.

    Error Handling:
        - ImportError: onnxruntime not installed -> logs critical, continues.
        - RuntimeError: model file missing -> logs error with setup instructions.
        - Other exceptions: logged, app continues without embeddings.
    """
    start = time.time()

    # Lazy import to avoid DLL conflicts during module loading
    onnx_available, init_onnx = _lazy_import_onnx()

    # Handle missing onnxruntime
    if not onnx_available or init_onnx is None:
        logger.warning(
            "Embedding model not initialized: onnxruntime unavailable. "
            "Basic features will work, but embedding-based features are disabled."
        )
        return None

    try:
        logger.info(f"Initializing Embedding Model: {MODEL_NAME} (ONNX Optimized)")

        # Initialize the optimized model
        # This handles caching, validation, and ONNX session creation internally
        model = init_onnx()

        duration = time.time() - start

        # Validate dimension (Sanity Check)
        if hasattr(model, "embed_dim") and model.embed_dim != EXPECTED_DIM:
            logger.error(
                f"Dimension mismatch: got {model.embed_dim}, expected {EXPECTED_DIM}"
            )

        # Log Memory Usage
        try:
            import psutil

            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            logger.info(
                f"Embedding Ready: {duration:.2f}s | Dim: {EXPECTED_DIM} | RAM: {mem_mb:.1f}MB"
            )
        except ImportError:
            logger.info(f"Embedding Ready: {duration:.2f}s | Dim: {EXPECTED_DIM}")

        return model

    except RuntimeError as e:
        # Model file missing - provide helpful instructions
        logger.error(
            f"Embedding model file missing: {e}. "
            f"Run 'python scripts/setup_onnx.py' to download and convert the model."
        )
        return None

    except Exception as e:
        # Unexpected error - log and continue
        logger.exception(f"Failed to initialize embedding model: {e}")
        return None


def get_embedding_model(app: FastAPI) -> Optional[Any]:
    """Retrieve the initialized embedding model from `app.state`.

    Args:
        app: FastAPI application instance.

    Returns:
        The embedding model if initialized, None otherwise.

    Raises:
        RuntimeError: If called before init_embedding or if model failed to load.
    """
    if not hasattr(app.state, "embedding_model"):
        raise RuntimeError(
            "Embedding model not initialized. Call init_embedding on startup."
        )

    model = app.state.embedding_model
    if model is None:
        logger.warning(
            "Embedding model is None - features requiring embeddings will fail"
        )

    return model
