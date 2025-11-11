"""Minimal FastAPI app for Necthrall Lite MVP.

Exposes a /health endpoint for readiness checks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from datetime import datetime
import sys

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = FastAPI(
    title="Necthrall Lite API",
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


@app.on_event("startup")
async def startup_event():
    """Validate configuration at startup"""
    try:
        import config  # Triggers validation

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


if __name__ == "__main__":
    # Keep local run simple; uvicorn invocation is optional in dev
    try:
        import uvicorn

        uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
    except Exception:
        logger.debug("uvicorn not available or failed to start (fine for tests)")
