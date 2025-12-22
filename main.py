"""Necthrall - AI-Powered Scientific Research Assistant.

FastAPI + NiceGUI integrated application for research paper analysis.
"""

from config import _threading
import os
import sys
import asyncio

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
    except (ImportError, OSError):
        pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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


async def search_worker():
    """Background worker to process search queries sequentially."""
    logger.info("Search worker started")
    while True:
        try:
            # Get a "work item" out of the queue.
            query, deep_mode, progress_callback, future = (
                await app.state.search_queue.get()
            )

            try:
                # Process the query
                result = await app.state.query_service.process_query(
                    query,
                    deep_mode=deep_mode,
                    progress_callback=progress_callback,
                )
                # Set the result
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                logger.exception("Error in search worker")
                if not future.done():
                    future.set_exception(e)
            finally:
                # Notify the queue that the "work item" has been processed.
                app.state.search_queue.task_done()

                # Rate limiting: sleep for 1 second
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Search worker cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in search worker loop: {e}")
            await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    """Initialize query service at startup."""
    try:
        import config

        from services.query_service import QueryService

        # Initialize embedding model if available
        embedding_loaded = False
        embedding_model = None
        try:
            from config.embedding_config import init_embedding

            await init_embedding(app)
            if hasattr(app.state, "embedding_model"):
                embedding_model = app.state.embedding_model
                embedding_loaded = True
                logger.info("✅ Embedding model loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Embedding config module not found: {e}")
        except RuntimeError as e:
            logger.warning(f"⚠️ ONNX model file missing: {e}")
        except Exception:
            logger.exception("⚠️ Embedding model failed to initialize")

        # Initialize QueryService with embedding model
        app.state.query_service = QueryService(embedding_model=embedding_model)
        logger.info("🚀 Query service initialized")

        # Initialize search queue and worker
        app.state.search_queue = asyncio.Queue()
        # --- START 3 WORKERS (3 Lanes) ---
        for i in range(1):
            asyncio.create_task(search_worker())

        logger.info("🚀 1 Search workers initialized (Parallel Lanes)")

        if embedding_loaded:
            logger.info("✅ Embedding model injected into QueryService")

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


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Process a research query."""
    if not hasattr(app.state, "query_service"):
        raise HTTPException(status_code=503, detail="Query service not initialized")

    try:
        result = await app.state.query_service.process_query(request.query)

        if not result.success:
            # Handle specific error cases
            if (
                result.error_message
                and "Semantic Scholar API is currently unavailable"
                in result.error_message
            ):
                raise HTTPException(
                    status_code=503,
                    detail=result.error_message,
                )

            # Include stage in error detail for debugging
            error_detail = (
                result.error_message or "Pipeline failed without error message"
            )
            if result.error_stage:
                error_detail = f"{error_detail} (Stage: {result.error_stage})"

            raise HTTPException(
                status_code=500,
                detail=error_detail,
            )

        # Format citations with IDs
        citations = []
        if result.passages:
            for idx, p in enumerate(result.passages, 1):
                # Handle both dict and object access
                text = p.get("text") if isinstance(p, dict) else getattr(p, "text", "")
                metadata = (
                    p.get("metadata")
                    if isinstance(p, dict)
                    else getattr(p, "metadata", {})
                ).copy()

                # Ensure score is in metadata if available
                score = (
                    p.get("score") if isinstance(p, dict) else getattr(p, "score", None)
                )
                if score is not None:
                    metadata["score"] = score

                citations.append({"id": idx, "text": text, "metadata": metadata})

        return {
            "query": request.query,
            "answer": result.answer
            or "No answer could be generated from the available sources.",
            "citations": citations,
            "finalists": result.finalists,
            "execution_time": result.execution_time,
            "timing_breakdown": result.timing_breakdown,
            "optimized_queries": result.optimized_queries,
            "quality_gate": result.quality_gate,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


# =============================================================================
# NiceGUI Frontend
# =============================================================================

from ui.pages import init_ui

# Initialize UI with FastAPI app reference
init_ui(app)

# Mount NiceGUI with FastAPI
ui.run_with(
    app,
    title="Nechtrall - Science, Distilled.",
    favicon="logo/favicon.png",
    dark=False,
    reconnect_timeout=30.0,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, log_level="info", reload=False)
