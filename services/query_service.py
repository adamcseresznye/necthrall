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
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from agents.query_optimization_agent import QueryOptimizationAgent
from config.config import Settings
from models.state import Paper, Passage
from services.discovery_service import DiscoveryService
from services.ingestion_service import IngestionService
from services.rag_service import RAGService
from services.research_service import ResearchService


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
        self.research_service = ResearchService(
            self.discovery_service,
            self.ingestion_service,
            self.rag_service,
        )

    async def close(self):
        await self.discovery_service.close()

    async def process_query(
        self,
        query: str,
        deep_mode: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """Process a user query via the multi-round ResearchService loop.

        Args:
            query: The user query string.
            deep_mode: Reserved for future use; ResearchService always uses PDF ingestion.
            progress_callback: Optional async callback to report progress.

        Returns:
            PipelineResult with synthesized answer and timing metadata.
        """
        start_time = time.perf_counter()

        async def report_progress():
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback()
                else:
                    progress_callback()

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

        try:
            logger.info(
                "QueryService: delegating to ResearchService for: {}", query[:100]
            )
            await report_progress()

            research_result = await self.research_service.research(query)

            # Map ResearchResult → PipelineResult (PipelineResult shape is unchanged)
            result.answer = research_result.answer
            result.optimized_queries = {
                "research_brief": research_result.research_brief,
                "searched_queries": research_result.searched_queries,
            }
            result.timing_breakdown = research_result.timing_breakdown
            result.refinement_count = research_result.rounds_completed
            result.execution_time = time.perf_counter() - start_time
            result.success = research_result.answer is not None

            logger.info(
                "QueryService: completed {} round(s) in {:.3f}s — answer={}",
                research_result.rounds_completed,
                result.execution_time,
                "yes" if result.answer else "no",
            )
            return result

        except Exception as e:
            logger.exception("QueryService: unexpected error in process_query")
            result.execution_time = time.perf_counter() - start_time
            result.success = False
            result.error_message = f"Unexpected error: {str(e)}"
            result.error_stage = "research_service"
            return result
