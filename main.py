from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid
import time
import json
import asyncio
from loguru import logger
from dotenv import load_dotenv

from utils.llm_client import LLMClient
from models.state import State
from orchestrator.graph import build_workflow

# Import for embedding model
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.warning(f"Failed to import sentence-transformers: {e}")
    SentenceTransformer = None
except OSError as e:
    logger.warning(
        f"Failed to load sentence-transformers (likely PyTorch DLL issue on Windows): {e}"
    )
    SentenceTransformer = None

# Load environment variables from .env file
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Necthrall Lite MVP",
    description="AI-powered scientific research assistant",
    version="0.1.0",
)


# --- Middleware ---
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Enforce 20-second hard timeout on all requests"""
    try:
        return await asyncio.wait_for(call_next(request), timeout=20.0)
    except asyncio.TimeoutError:
        request_id = request.headers.get("X-Request-ID", "unknown")
        logger.error(
            json.dumps(
                {
                    "event": "request_timeout",
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "user_agent": request.headers.get("user-agent"),
                    "client_ip": request.client.host,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        return JSONResponse(
            status_code=408,
            content={
                "error": "Request timeout",
                "message": "Query took longer than 20 seconds. Please try a more specific query.",
            },
        )


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    max_papers: int = Field(25, ge=1, le=100)


class QueryResponse(BaseModel):
    request_id: str
    query: str
    test_llm_response: str
    model_used: str
    execution_time: float
    status: str


class HealthResponse(BaseModel):
    status: str
    embedding_model_loaded: bool
    model_name: str | None
    embedding_dimension: int | None
    gemini_status: str
    groq_status: str
    overall_status: str


# --- Global Clients ---
llm_client = LLMClient()
# workflow = build_workflow()  # Moved to per-request initialization

# --- Structured Logging Configuration ---
logger.add(
    "logs/necthrall_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{message}",
    serialize=True,
)


# --- FastAPI Startup Event Handler ---
@app.on_event("startup")
async def load_embedding_model():
    """
    Load SentenceTransformer embedding model at startup and cache in app.state.

    This eliminates the 30-second cold-start penalty on first query by pre-loading
    the all-MiniLM-L6-v2 model (~90MB) during application initialization.

    Model is stored in app.state for global access by FilteringAgent and ProcessingAgent.
    """
    if SentenceTransformer is None:
        logger.warning(
            "sentence-transformers not available, skipping embedding model loading"
        )
        app.state.embedding_model = None
        app.state.cross_encoder_model = None  # Also set cross-encoder to None
        return

    start_time = time.time()
    logger.info("Loading SentenceTransformer embedding model (all-MiniLM-L6-v2)...")

    try:
        # Load embedding model (downloads from HuggingFace on first run, caches locally afterward)
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        app.state.embedding_model = embedding_model
        _ = embedding_model.encode(
            ["test sentence"], show_progress_bar=False
        )  # Warm up
        elapsed_embed = time.time() - start_time
        logger.info(f"✓ Embedding model loaded successfully in {elapsed_embed:.2f}s")

        # Load CrossEncoder model
        from sentence_transformers import CrossEncoder

        cross_encoder_start_time = time.time()
        logger.info(
            "Loading CrossEncoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)..."
        )
        cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        app.state.cross_encoder_model = cross_encoder_model
        _ = cross_encoder_model.predict(
            [("test query", "test passage")], show_progress_bar=False
        )  # Warm up
        elapsed_cross = time.time() - cross_encoder_start_time
        logger.info(f"✓ CrossEncoder model loaded successfully in {elapsed_cross:.2f}s")

    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")
        app.state.embedding_model = None
        app.state.cross_encoder_model = None


@app.on_event("shutdown")
async def cleanup_resources():
    """Cleanup resources on application shutdown"""
    if hasattr(app.state, "embedding_model") and app.state.embedding_model is not None:
        logger.info("Cleaning up embedding model...")
        del app.state.embedding_model


# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def process_query(request_data: QueryRequest, request: Request):
    """
    Accepts a user's scientific query, tests LLM connectivity, and initiates the LangGraph workflow.
    This Day 1 endpoint is primarily for testing the foundational plumbing.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request_size = int(request.headers.get("content-length", 0))

    logger.info(
        json.dumps(
            {
                "event": "query_received",
                "request_id": request_id,
                "query": request_data.query,
                "max_papers": request_data.max_papers,
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host,
                "request_size": request_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    )

    try:
        # 1. Initialize State
        logger.info(
            json.dumps({"event": "query_processing_start", "request_id": request_id})
        )
        initial_state = State(
            request_id=request_id,
            query=request_data.query,
            max_papers=request_data.max_papers,
        )

        # 2. Test LLM Connectivity
        logger.info(
            json.dumps(
                {
                    "event": "llm_call",
                    "request_id": request_id,
                    "prompt": f"Provide a one-sentence overview of this query: {request_data.query}",
                }
            )
        )
        llm_test_response = llm_client.generate(
            messages=[
                {
                    "role": "user",
                    "content": f"Provide a one-sentence overview of this query: {request_data.query}",
                }
            ],
            max_tokens=100,
        )

        # 3. Invoke LangGraph Workflow
        logger.info(
            json.dumps({"event": "workflow_execution", "request_id": request_id})
        )
        workflow = build_workflow(request, None)  # None for mock_agents parameter
        result = workflow.invoke(initial_state)

        execution_time = time.time() - start_time
        logger.info(
            json.dumps(
                {
                    "event": "query_processing_complete",
                    "request_id": request_id,
                    "execution_time": execution_time,
                }
            )
        )

        return QueryResponse(
            request_id=request_id,
            query=request_data.query,
            test_llm_response=llm_test_response["content"],
            model_used=llm_test_response["model_used"],
            execution_time=execution_time,
            status="day1_test_success",
        )

    except Exception as e:
        logger.error(
            json.dumps(
                {
                    "event": "query_processing_error",
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        if "Both primary (Gemini) and fallback (Groq) LLM providers failed" in str(e):
            raise HTTPException(status_code=500, detail="LLM service unavailable")
        else:
            raise HTTPException(
                status_code=500, detail="An internal server error occurred."
            )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint that verifies embedding model is loaded.

    Returns:
        - status: "healthy" if model loaded, "degraded" if not
        - embedding_model_loaded: Boolean indicating model availability
        - model_name: Name of loaded model
        - embedding_dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2)
    """
    gemini_status = "ok"
    groq_status = "ok"

    try:
        await asyncio.to_thread(
            llm_client.primary_llm.invoke, [{"role": "user", "content": "Health check"}]
        )
    except Exception:
        gemini_status = "error"

    try:
        await asyncio.to_thread(
            llm_client.fallback_llm.invoke,
            [{"role": "user", "content": "Health check"}],
        )
    except Exception:
        groq_status = "error"

    # Check embedding model status
    model_loaded = (
        hasattr(app.state, "embedding_model") and app.state.embedding_model is not None
    )

    if model_loaded:
        model = app.state.embedding_model
        return HealthResponse(
            status="healthy",
            embedding_model_loaded=True,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,  # all-MiniLM-L6-v2 produces 384-dim vectors
            gemini_status=gemini_status,
            groq_status=groq_status,
            overall_status=(
                "ok" if gemini_status == "ok" and groq_status == "ok" else "degraded"
            ),
        )
    else:
        return HealthResponse(
            status="degraded",
            embedding_model_loaded=False,
            model_name=None,
            embedding_dimension=None,
            gemini_status=gemini_status,
            groq_status=groq_status,
            overall_status="degraded",
        )
