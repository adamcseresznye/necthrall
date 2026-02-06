"""Necthrall - AI-Powered Scientific Research Assistant.

FastAPI + NiceGUI integrated application for research paper analysis.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from config.config import get_settings

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from nicegui import ui

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
    lifespan=lifespan,
)

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
    storage_secret=get_settings().NICEGUI_STORAGE_SECRET,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, log_level="info", reload=False)
