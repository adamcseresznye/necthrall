"""Necthrall - AI-Powered Scientific Research Assistant.

FastAPI + NiceGUI integrated application for research paper analysis.
"""

import os
import sys

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# WINDOWS DLL FIX: Import torch FIRST before any ONNX-related imports
# This prevents DLL initialization errors with c10.dll when loading CrossEncoder
# See docs/torch_dll_fix.md for details
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

    try:
        import torch  # noqa: F401 - Must be imported before ONNX
    except ImportError:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from datetime import datetime
from nicegui import ui

# Configure loguru with immediate flush for Windows compatibility
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = FastAPI(
    title="Necthrall API",
    description="AI-powered scientific research assistant",
    version="3.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize query service at startup."""
    try:
        import config

        from services.query_service import QueryService

        app.state.query_service = QueryService()
        logger.info("🚀 Query service initialized")

        # Initialize embedding model if available
        embedding_loaded = False
        try:
            from config.embedding_config import init_embedding

            await init_embedding(app)
            embedding_loaded = True
            logger.info("✅ Embedding model loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Embedding config module not found: {e}")
        except RuntimeError as e:
            logger.warning(f"⚠️ ONNX model file missing: {e}")
        except Exception:
            logger.exception("⚠️ Embedding model failed to initialize")

        # Inject embedding model into QueryService if loaded
        if embedding_loaded and hasattr(app.state, "embedding_model"):
            try:
                app.state.query_service.embedding_model = app.state.embedding_model
                logger.info("✅ Embedding model injected into QueryService")
            except Exception as e:
                logger.warning(f"❌ Failed to inject embedding model: {e}")

        logger.info("✅ Application startup successful")
        logger.info(f"Query optimization model: {config.QUERY_OPTIMIZATION_MODEL}")
        logger.info(f"Synthesis model: {config.SYNTHESIS_MODEL}")
    except Exception as e:
        logger.exception("❌ Configuration validation failed")
        sys.exit(1)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
    }


# =============================================================================
# NiceGUI Frontend
# =============================================================================

from ui.pages import init_ui

# Initialize UI with FastAPI app reference
init_ui(app)

# Mount NiceGUI with FastAPI
ui.run_with(
    app,
    title="Nechtrall",
    favicon="logo/favicon.png",
    dark=False,
    reconnect_timeout=30.0,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, log_level="info", reload=True)
