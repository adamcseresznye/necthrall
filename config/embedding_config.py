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
"""

from __future__ import annotations

import time
from typing import Any
from fastapi import FastAPI
from loguru import logger

# Import the optimized ONNX model
from config.onnx_embedding import initialize_embedding_model as init_onnx

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384


async def init_embedding(app: FastAPI) -> None:
    """Initialize ONNX Embedding Model and store on app state.

    Raises:
        RuntimeError: If model initialization fails.
    """
    start = time.time()

    try:
        logger.info(f"Initializing Embedding Model: {MODEL_NAME} (ONNX Optimized)")

        # Initialize the optimized model
        # This handles caching, validation, and ONNX session creation internally
        model = init_onnx()

        # Store on app state
        app.state.embedding_model = model

    except Exception as e:
        logger.exception("Failed to initialize embedding model")
        raise RuntimeError("Embedding model initialization failed") from e

    duration = time.time() - start

    # Validate dimension (Sanity Check)
    if hasattr(model, "embed_dim") and model.embed_dim != EXPECTED_DIM:
        raise ValueError(
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


def get_embedding_model(app: FastAPI) -> Any:
    """Retrieve the initialized embedding model from `app.state`."""
    model = getattr(app.state, "embedding_model", None)
    if model is None:
        raise RuntimeError(
            "Embedding model not initialized. Call init_embedding on startup."
        )
    return model
