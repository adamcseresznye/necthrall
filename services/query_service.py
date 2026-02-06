"""Query service for orchestrating the pipeline.

Handles the complete query processing pipeline with comprehensive error handling,
detailed timing instrumentation, and structured logging.

Stages (1-4):
    1. Query Optimization
    2. Semantic Scholar Search
    3. Quality Gate
    4. Composite Scoring

Stages (5-8):
    5. PDF Acquisition
    6. Processing & Embedding
    7. Hybrid Retrieval
    8. Cross-Encoder Reranking
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from loguru import logger

from agents.query_optimization_agent import QueryOptimizationAgent
from config.config import Settings
from models.state import Paper, Passage
from services.discovery_service import DiscoveryService
from services.exceptions import QueryServiceError
from services.ingestion_service import IngestionService
from services.rag_service import RAGService


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        query: Original user query.
        optimized_queries: Dict with primary, broad, alternative, and final_rephrase.
        quality_gate: Quality gate validation results.
        finalists: Top ranked papers after composite scoring.
        execution_time: Total pipeline execution time in seconds.
        timing_breakdown: Per-stage timing in seconds.
        success: Whether pipeline completed successfully.
        error_message: Error message if pipeline failed.
        error_stage: Stage where error occurred.
        passages: Top ranked passages after reranking.
        answer: Synthesized answer from passages (Stage 9).
        citation_verification: Citation verification result (Stage 10).
        refinement_count: Number of query refinement attempts (0 or 1).
    """

    query: str
    optimized_queries: Dict[str, Any]
    quality_gate: Dict[str, Any]
    finalists: List[Paper]
    execution_time: float
    timing_breakdown: Dict[str, float]
    success: bool
    error_message: Optional[str] = None
    error_stage: Optional[str] = None
    passages: List[Passage] = field(default_factory=list)
    answer: Optional[str] = None
    citation_verification: Optional[Dict[str, Any]] = None
    refinement_count: int = 0


class QueryService:
    """Service for orchestrating the query pipeline.
    Provides comprehensive error handling and performance monitoring.
    """

    def __init__(self, settings: Settings, embedding_model: Any):
        """Initialize the query service.

        Args:
            settings: Application settings.
            embedding_model: Pre-loaded embedding model for query/passage embedding.
        """
        self.settings = settings
        self.embedding_model = embedding_model

        # Initialize agents/services immediately
        self.optimization_agent = QueryOptimizationAgent()
        self.discovery_service = DiscoveryService(settings)
        self.ingestion_service = IngestionService(embedding_model)
        self.rag_service = RAGService(embedding_model, settings)

    async def close(self):
        await self.discovery_service.close()

    async def process_query(
        self,
        query: str,
        deep_mode: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """Process a user query through the complete pipeline.

        Stages:
            1-4: Discovery (Optimization, Search, Quality Gate, Ranking)
            5-6: Ingestion (Acquisition, Processing)
            7-10: RAG (Retrieval, Reranking, Synthesis, Verification)

        Args:
            query: The user query string
            deep_mode: Whether to perform deep PDF analysis (True) or fast abstract search (False)
            progress_callback: Optional async callback to report progress

        Returns:
            PipelineResult with all processing results and metadata

        Raises:
            QueryServiceError: For pipeline-specific errors
        """
        start_time = time.perf_counter()
        timing_breakdown = {}

        # Initialize result with defaults
        result = PipelineResult(
            query=query,
            optimized_queries={},
            quality_gate={},
            finalists=[],
            execution_time=0.0,
            timing_breakdown={},
            success=False,
            passages=[],
            refinement_count=0,
        )

        # Helper to safely await callback
        async def report_progress():
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback()
                else:
                    progress_callback()

        try:
            logger.info("Starting query pipeline for: {}", query[:100])
            await report_progress()

            # ======0: Planning (Optimization)
            # =====================================================================
            opt_start = time.perf_counter()
            optimization_result = None
            try:
                optimization_result = (
                    await self.optimization_agent.generate_dual_queries(query)
                )
            except Exception as e:
                logger.error(
                    "Optimization failed in QueryService, proceeding with fallback: {}",
                    e,
                )

            opt_time = time.perf_counter() - opt_start

            # =====================================================================
            # Phase 1: Discovery (Stages 1-4)
            # =====================================================================
            discovery_result = await self.discovery_service.discover(
                query, optimized_data=optimization_result
            )

            # Update result and timing
            result.optimized_queries = discovery_result.optimized_queries
            result.quality_gate = discovery_result.quality_gate
            result.finalists = discovery_result.finalists
            result.refinement_count = discovery_result.refinement_count
            timing_breakdown.update(discovery_result.timing_breakdown)

            # Ensure our optimization time is preserved if we ran it
            if optimization_result:
                timing_breakdown["query_optimization"] = opt_time

            # Check if we have finalists to process
            if not discovery_result.finalists:
                logger.info("ℹ️ No finalists to process")
                result.execution_time = time.perf_counter() - start_time
                result.timing_breakdown = timing_breakdown
                result.success = True
                return result

            # =====================================================================
            # Phase 2: Ingestion (Stages 5-6)
            # =====================================================================
            await report_progress()

            chunks = []

            if deep_mode:
                ingestion_result = await self.ingestion_service.ingest(
                    discovery_result.finalists, query
                )

                # Update timing
                timing_breakdown.update(ingestion_result.timing_breakdown)
                chunks = ingestion_result.chunks
            else:
                chunks = await self.ingestion_service.ingest_abstracts(
                    discovery_result.finalists
                )

            # Check if we have chunks to process
            if not chunks:
                logger.warning("⚠️ No chunks generated from ingestion (or fast mode)")
                result.execution_time = time.perf_counter() - start_time
                result.timing_breakdown = timing_breakdown
                result.success = True
                return result

            # =====================================================================
            # Phase 3: RAG (Stages 7-10)
            # =====================================================================
            await report_progress()

            # Use final_rephrase for RAG if available, else original query
            rag_query = discovery_result.optimized_queries.get("final_rephrase", query)

            rag_result = await self.rag_service.answer(rag_query, chunks)

            # Update result and timing
            result.passages = rag_result.passages
            result.answer = rag_result.answer
            result.citation_verification = rag_result.citation_verification
            timing_breakdown.update(rag_result.timing_breakdown)

            # Finalize result
            result.execution_time = time.perf_counter() - start_time
            result.timing_breakdown = timing_breakdown
            result.success = True

            logger.info(
                "✅ Pipeline completed successfully in {:.3f}s", result.execution_time
            )
            return result

        except QueryServiceError as e:
            logger.error("Pipeline failed at stage {}: {}", e.stage, e.message)
            result.execution_time = time.perf_counter() - start_time
            result.timing_breakdown = timing_breakdown
            result.success = False
            result.error_message = e.message
            result.error_stage = e.stage
            return result

        except Exception as e:
            logger.exception("Unexpected pipeline error")
            result.execution_time = time.perf_counter() - start_time
            result.timing_breakdown = timing_breakdown
            result.success = False
            result.error_message = f"Unexpected error: {str(e)}"
            result.error_stage = "unknown"
            return result
