"""Minimal FastAPI app for Necthrall Lite MVP.

Exposes a /health endpoint for readiness checks.
"""

# CRITICAL: Import torch FIRST to avoid DLL initialization errors on Windows
# See: https://github.com/pytorch/pytorch/issues/91966
import os


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


try:
    import torch
except ImportError:
    pass  # torch may not be installed yet

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime
import sys
import time
import numpy as np
from services.query_service import (
    QueryOptimizationError,
    SemanticScholarError,
    QualityGateError,
    RankingError,
)

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
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


@app.on_event("startup")
async def startup_event():
    """Validate configuration at startup

    Week 2: Will add local embedding model for passage-level chunking.
    """
    try:
        import config  # Triggers validation

        # Initialize query service (no embedding model needed for Week 1 behavior)
        from services.query_service import QueryService

        app.state.query_service = QueryService()

        # Initialize the local embedding model for Week 2+ pipelines
        try:
            from config.embedding_config import init_embedding

            await init_embedding(app)
        except Exception:
            logger.exception(
                "Embedding model failed to initialize; continuing without it"
            )

        logger.info("Query service initialized")
        logger.info("Application startup successful")
        logger.info(f"Query optimization model: {config.QUERY_OPTIMIZATION_MODEL}")
        logger.info(f"Synthesis model: {config.SYNTHESIS_MODEL}")
    except Exception as e:
        logger.exception("Configuration validation failed")
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
    """Execute the Week 1 pipeline: query_optimization → semantic_scholar_search → quality_gate → composite_scoring.

    Accepts a user query and returns top 5-10 ranked papers with execution timing and quality metrics.
    """
    try:
        # Use the query service to process the request
        result = await app.state.query_service.process_query(request.query)

        return QueryResponse(
            query=result.query,
            optimized_queries=result.optimized_queries,
            quality_gate=result.quality_gate,
            finalists=result.finalists,
            execution_time=result.execution_time,
            timing_breakdown=result.timing_breakdown,
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
