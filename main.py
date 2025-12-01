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
    SynthesisError,
    VerificationError,
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
    """Request model for /query endpoint.

    Attributes:
        query: The user's scientific research question.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User query string",
        json_schema_extra={
            "example": "What are the cardiovascular effects of intermittent fasting?"
        },
    )


class CitationItem(BaseModel):
    """A single citation with text content and metadata for frontend display.

    Attributes:
        id: 1-based citation index corresponding to [N] in the answer.
        text: The passage text content.
        metadata: Additional metadata (paper_id, paper_title, section, score, etc.).
    """

    id: int = Field(
        ..., description="1-based citation index corresponding to [N] in the answer"
    )
    text: str = Field(..., description="The passage text content")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (paper_id, paper_title, section, score)",
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint.

    This is the CRITICAL schema for the Frontend (Week 4).

    Attributes:
        answer: The synthesized answer text with inline [N] citations.
        citations: List of citation items for frontend modal display.
        finalists: List of papers found during search.
        execution_time: Total pipeline execution time in seconds.
        timing_breakdown: Per-stage timing in seconds.
    """

    answer: str = Field(
        ...,
        description="The synthesized answer text with inline [N] citations",
    )
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="List of citations with id, text, and metadata for frontend modal",
    )
    finalists: list = Field(
        default_factory=list,
        description="List of papers found during search",
    )
    execution_time: float = Field(
        ...,
        description="Total pipeline execution time in seconds",
    )
    timing_breakdown: dict = Field(
        default_factory=dict,
        description="Per-stage timing breakdown in seconds",
    )


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
    """Execute the full 10-stage pipeline and return a synthesized answer with citations.

    **Pipeline Stages:**
    - Week 1 (1-4): query_optimization → semantic_scholar_search → quality_gate → composite_scoring
    - Week 2 (5-8): pdf_acquisition → processing → hybrid_retrieval → cross_encoder_reranking
    - Week 3 (9-10): synthesis → citation_verification

    **Returns:**
    - `answer`: Synthesized text with inline [N] citations
    - `citations`: List of passage objects with id, text, and metadata for frontend modals
    - `finalists`: Papers found during search
    - `execution_time`: Total processing time in seconds
    - `timing_breakdown`: Per-stage timing breakdown

    **Example Response:**
    ```json
    {
        "answer": "Intermittent fasting reduces blood pressure [1] and improves cardiac function [2].",
        "citations": [
            {"id": 1, "text": "Blood pressure decreased by 8.5 mmHg...", "metadata": {...}},
            {"id": 2, "text": "Cardiac function improved significantly...", "metadata": {...}}
        ],
        "finalists": [...],
        "execution_time": 12.5,
        "timing_breakdown": {"query_optimization": 0.5, ...}
    }
    ```
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        # Use the query service to process the request
        result = await app.state.query_service.process_query(request.query)

        # Check if pipeline succeeded
        if not result.success:
            logger.error(
                f"Pipeline failed at stage '{result.error_stage}': {result.error_message}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed at stage '{result.error_stage}': {result.error_message}",
            )

        # Transform passages to citations with id, text, and metadata
        citations = []
        for idx, passage in enumerate(result.passages):
            try:
                # Extract text content
                text = (
                    passage.node.get_content()
                    if hasattr(passage.node, "get_content")
                    else str(passage.node)
                )

                # Extract metadata
                metadata = (
                    passage.node.metadata.copy()
                    if hasattr(passage.node, "metadata") and passage.node.metadata
                    else {}
                )

                # Add score to metadata for frontend use
                metadata["score"] = passage.score

                citation_item = CitationItem(
                    id=idx + 1,  # 1-based index to match [N] citations
                    text=text,
                    metadata=metadata,
                )
                citations.append(citation_item)
            except Exception as e:
                logger.warning(f"Failed to serialize passage {idx}: {e}")
                continue

        # Get the answer (fallback to empty string if None)
        answer = (
            result.answer or "No answer could be generated from the available sources."
        )

        logger.info(
            f"Query processed successfully: {len(citations)} citations, "
            f"{len(result.finalists)} finalists, {result.execution_time:.2f}s"
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            finalists=result.finalists,
            execution_time=result.execution_time,
            timing_breakdown=result.timing_breakdown,
        )

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
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
    except AcquisitionError as e:
        logger.exception("PDF acquisition error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except ProcessingError as e:
        logger.exception("Processing error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except RetrievalError as e:
        logger.exception("Retrieval error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except RerankingError as e:
        logger.exception("Reranking error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except SynthesisError as e:
        logger.exception("Synthesis error for query: {}", request.query[:100])
        raise HTTPException(status_code=e.http_status, detail=e.message)
    except VerificationError as e:
        logger.exception("Verification error for query: {}", request.query[:100])
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
