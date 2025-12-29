"""Necthrall - AI-Powered Scientific Research Assistant.

FastAPI + NiceGUI integrated application for research paper analysis.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from config import _threading

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from nicegui import ui
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Global concurrency control
MAX_CONCURRENT_SEARCHES = 2
search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)


# Move search_worker definition BEFORE app instantiation so it can be used in lifespan
async def search_worker(app_instance: FastAPI):
    """Background worker to process search queries sequentially."""
    logger.info("Search worker started")
    while True:
        try:
            # Get a "work item" out of the queue.
            # Using the passed instance instead of global 'app' for better safety
            query, deep_mode, progress_callback, future = (
                await app_instance.state.search_queue.get()
            )

            try:
                # Process the query
                result = await app_instance.state.query_service.process_query(
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
                app_instance.state.search_queue.task_done()

                # Rate limiting: sleep for 1 second
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Search worker cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in search worker loop: {e}")
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown logic.
    Replaces @app.on_event("startup").
    """

    # Initialize state attributes to avoid AttributeErrors if startup fails
    app.state.search_queue = asyncio.Queue()
    app.state.query_service = None
    app.state.embedding_model = None
    app.state.workers = []

    # --- STARTUP LOGIC ---
    try:
        from config.config import get_settings
        from config.embedding_config import init_embedding
        from services.query_service import QueryService

        # 1. Load Settings
        settings = get_settings()
        settings.validate_keys()

        # 2. Init Embedding Model (puts it on app.state.embedding_model)
        embedding_model = None
        try:
            embedding_model = await init_embedding()
            app.state.embedding_model = embedding_model
            if embedding_model:
                logger.info("✅ Embedding model loaded successfully")
            else:
                logger.warning("⚠️ Embedding model failed to load")
        except Exception as e:
            logger.warning(f"⚠️ Embedding model initialization failed: {e}")
            # app.state.embedding_model is already None

        # 3. Create Singleton QueryService
        # This will internally create Discovery, Ingestion, and RAG services
        query_service = QueryService(settings=settings, embedding_model=embedding_model)

        # 4. Store in State
        app.state.query_service = query_service
        logger.info("🚀 Query service initialized")

        # 5. Init Workers
        # Start N workers for better concurrency
        CONCURRENT_WORKERS: int = 1
        app.state.workers = [
            asyncio.create_task(search_worker(app)) for _ in range(CONCURRENT_WORKERS)
        ]

        logger.info(
            f"🚀 {CONCURRENT_WORKERS} Search workers initialized (Parallel Lanes)"
        )
        logger.info("✅ Application startup successful")
        logger.info(f"Query optimization model: {settings.QUERY_OPTIMIZATION_MODEL}")
        logger.info(f"Synthesis model: {settings.SYNTHESIS_MODEL}")

    except Exception as e:
        logger.critical(f"Failed to initialize application: {e}")
        # app.state.query_service is already None

    # Expose concurrency controls to UI
    app.state.search_semaphore = search_semaphore
    app.state.max_concurrent_searches = MAX_CONCURRENT_SEARCHES

    yield  # Application runs here

    # --- SHUTDOWN LOGIC ---
    logger.info("Shutting down Necthrall...")

    # Cancel workers
    if hasattr(app.state, "workers"):
        logger.info("🛑 Shutting down: Cancelling workers...")
        for worker in app.state.workers:
            worker.cancel()
        await asyncio.gather(*app.state.workers, return_exceptions=True)

    # Close services
    if hasattr(app.state, "query_service") and app.state.query_service:
        await app.state.query_service.close()

    logger.info("Shutdown complete")


app = FastAPI(
    title="Necthrall API",
    description="AI-powered scientific research assistant",
    version="3.0.0",
    lifespan=lifespan,  # Register the lifespan handler here
)

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
@limiter.limit("5/hour")
async def query_endpoint(query_request: QueryRequest, request: Request):
    """Process a research query."""
    if not hasattr(app.state, "query_service"):
        raise HTTPException(status_code=503, detail="Query service not initialized")

    if search_semaphore.locked():
        raise HTTPException(
            status_code=503, detail="Server is busy. Please try again later."
        )

    async with search_semaphore:
        try:
            result = await app.state.query_service.process_query(query_request.query)

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
                    text = (
                        p.get("text") if isinstance(p, dict) else getattr(p, "text", "")
                    )
                    metadata = (
                        p.get("metadata")
                        if isinstance(p, dict)
                        else getattr(p, "metadata", {})
                    ).copy()

                    # Ensure score is in metadata if available
                    score = (
                        p.get("score")
                        if isinstance(p, dict)
                        else getattr(p, "score", None)
                    )
                    if score is not None:
                        metadata["score"] = score

                    citations.append({"id": idx, "text": text, "metadata": metadata})

            return {
                "query": query_request.query,
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
