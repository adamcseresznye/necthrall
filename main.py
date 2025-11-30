"""Minimal FastAPI app for Necthrall Lite MVP.

Exposes a /health endpoint for readiness checks.
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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime
import time
import numpy as np
from services.query_service import (
    QueryOptimizationError,
    SemanticScholarError,
    QualityGateError,
    RankingError,
    AcquisitionError,
    ProcessingError,
    RetrievalError,
    RerankingError,
)

# Configure loguru with immediate flush for Windows compatibility
logger.remove()
logger.add(
    sys.stderr,  # Use stderr for immediate output (not buffered like stdout)
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = FastAPI(
    title="Necthrall API",
    description="AI-powered scientific research assistant",
    version="3.0.0-mvp",
)

# Configure CORS (permissive for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will tighten in Week 4
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="User query string"
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""

    query: str
    optimized_queries: dict
    quality_gate: dict
    finalists: list = Field(default_factory=list)
    execution_time: float
    timing_breakdown: dict
    passages: list = Field(default_factory=list)


@app.on_event("startup")
async def startup_event():
    """Validate configuration at startup

    Week 2: Will add local embedding model for passage-level chunking.
    """
    try:
        import config  # Triggers validation

        # Initialize query service first (no embedding model needed for Week 1 behavior)
        from services.query_service import QueryService

        app.state.query_service = QueryService()
        logger.info("🚀 Query service initialized (Week 1 pipeline ready)")

        # Initialize the local embedding model for Week 2+ pipelines
        embedding_loaded = False
        try:
            from config.embedding_config import init_embedding

            await init_embedding(app)
            embedding_loaded = True
            logger.info("✅ Embedding model loaded successfully")
        except ImportError as e:
            logger.warning(
                "⚠️ Embedding config module not found: {}. Week 1 pipeline still available.",
                str(e),
            )
        except RuntimeError as e:
            logger.warning(
                "⚠️ ONNX model file missing or runtime error: {}. Week 1 pipeline still available.",
                str(e),
            )
        except Exception:
            logger.exception(
                "⚠️ Embedding model failed to initialize; continuing without it (Week 1 pipeline available)"
            )

        # Inject embedding model into QueryService if successfully loaded
        if embedding_loaded and hasattr(app.state, "embedding_model"):
            try:
                app.state.query_service.embedding_model = app.state.embedding_model
                logger.info("✅ Embedding model injected into QueryService")
            except Exception as e:
                logger.warning(
                    "❌ Failed to inject embedding model into QueryService: {}. Week 1 pipeline still available.",
                    str(e),
                )
        elif not embedding_loaded:
            logger.info(
                "ℹ️ QueryService running without embedding model (Week 1 pipeline mode)"
            )

        logger.info("✅ Application startup successful")
        logger.info(f"Query optimization model: {config.QUERY_OPTIMIZATION_MODEL}")
        logger.info(f"Synthesis model: {config.SYNTHESIS_MODEL}")
    except Exception as e:
        logger.exception("❌ Configuration validation failed")
        sys.exit(1)


@app.get("/")
async def root():
    """Root endpoint with API documentation link"""
    return {
        "message": "Necthrall API v3.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0-mvp",
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Execute the full Week 1 + Week 2 pipeline.

    Week 1: query_optimization → semantic_scholar_search → quality_gate → composite_scoring
    Week 2: pdf_acquisition → processing → hybrid_retrieval → cross_encoder_reranking

    Accepts a user query and returns top 5-10 ranked papers with passages, execution timing, and quality metrics.
    """
    try:
        # Use the query service to process the request
        result = await app.state.query_service.process_query(request.query)

        # Convert NodeWithScore objects to serializable dicts for passages
        serialized_passages = []
        for passage in result.passages:
            try:
                passage_dict = {
                    "content": (
                        passage.node.get_content()
                        if hasattr(passage.node, "get_content")
                        else str(passage.node)
                    ),
                    "score": passage.score,
                    "metadata": (
                        passage.node.metadata
                        if hasattr(passage.node, "metadata")
                        else {}
                    ),
                }
                serialized_passages.append(passage_dict)
            except Exception as e:
                logger.warning(f"Failed to serialize passage: {e}")
                continue

        return QueryResponse(
            query=result.query,
            optimized_queries=result.optimized_queries,
            quality_gate=result.quality_gate,
            finalists=result.finalists,
            execution_time=result.execution_time,
            timing_breakdown=result.timing_breakdown,
            passages=serialized_passages,
        )

    except QueryOptimizationError as e:
        logger.exception("Query optimization error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except SemanticScholarError as e:
        logger.exception("Semantic Scholar error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except QualityGateError as e:
        logger.exception("Quality gate error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except RankingError as e:
        logger.exception("Ranking error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except Exception as e:
        logger.exception(
            "Unexpected error in query endpoint for query: {}", request.query[:100]
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )


if __name__ == "__main__":
    # Keep local run simple; uvicorn invocation is optional in dev
    try:
        import uvicorn

        uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
    except Exception:
        logger.debug("uvicorn not available or failed to start (fine for tests)")
