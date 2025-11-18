"""Query service for orchestrating the Week 1 pipeline.

Handles the complete query processing pipeline with comprehensive error handling,
detailed timing instrumentation, and structured logging.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger
import numpy as np

from agents.query_optimization_agent import QueryOptimizationAgent
from agents.semantic_scholar_client import SemanticScholarClient
from agents.quality_gate import validate_quality
from agents.ranking_agent import RankingAgent
import config


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    query: str
    optimized_queries: Dict[str, str]
    quality_gate: Dict[str, Any]
    finalists: List[Dict[str, Any]]
    execution_time: float
    timing_breakdown: Dict[str, float]
    success: bool
    error_message: Optional[str] = None
    error_stage: Optional[str] = None


class QueryServiceError(Exception):
    """Base exception for query service errors."""

    def __init__(self, message: str, stage: str, http_status: int = 500):
        self.message = message
        self.stage = stage
        self.http_status = http_status
        super().__init__(message)


class QueryOptimizationError(QueryServiceError):
    """Error during query optimization."""

    def __init__(self, message: str):
        super().__init__(message, "query_optimization", 500)


class SemanticScholarError(QueryServiceError):
    """Error during Semantic Scholar API calls."""

    def __init__(self, message: str):
        super().__init__(message, "semantic_scholar_search", 503)


class QualityGateError(QueryServiceError):
    """Error during quality gate validation."""

    def __init__(self, message: str):
        super().__init__(message, "quality_gate", 500)


class RankingError(QueryServiceError):
    """Error during paper ranking."""

    def __init__(self, message: str):
        super().__init__(message, "composite_scoring", 500)


class QueryService:
    """Service for orchestrating the Week 1 query pipeline.

    Handles query optimization, paper retrieval, quality validation, and ranking
    with comprehensive error handling and performance monitoring.
    """

    def __init__(self, embedding_model: Any = None):
        """Initialize the query service.

        Args:
            embedding_model: Pre-loaded SentenceTransformer model for query embedding.
                           Optional for Week 1 (uses Semantic Scholar API embeddings).
                           Required for Week 2+ (local passage embeddings).
        """
        self.embedding_model = embedding_model
        self._optimizer = None
        self._client = None
        self._ranker = None

    def _get_optimizer(self) -> QueryOptimizationAgent:
        """Lazy initialization of query optimizer."""
        if self._optimizer is None:
            self._optimizer = QueryOptimizationAgent()
        return self._optimizer

    def _get_client(self) -> SemanticScholarClient:
        """Lazy initialization of Semantic Scholar client."""
        if self._client is None:
            self._client = SemanticScholarClient(
                api_key=config.SEMANTIC_SCHOLAR_API_KEY
            )
        return self._client

    def _get_ranker(self) -> RankingAgent:
        """Lazy initialization of ranking agent."""
        if self._ranker is None:
            self._ranker = RankingAgent()
        return self._ranker

    async def process_query(self, query: str) -> PipelineResult:
        """Process a user query through the complete Week 1 pipeline.

        Args:
            query: The user query string

        Returns:
            PipelineResult with all processing results and metadata

        Raises:
            QueryServiceError: For pipeline-specific errors
        """
        start_time = time.perf_counter()
        timing_breakdown = {}
        result = PipelineResult(
            query=query,
            optimized_queries={},
            quality_gate={},
            finalists=[],
            execution_time=0.0,
            timing_breakdown={},
            success=False,
        )

        try:
            logger.info("Starting query pipeline for: {}", query[:100])

            # Stage 1: Query Optimization
            logger.info("Stage 1: Query optimization - input query: {}", query[:100])
            stage_start = time.perf_counter()
            try:
                optimized_queries = await self._get_optimizer().generate_dual_queries(
                    query
                )
                # Debug: log the optimized queries dict so we can verify whether
                # the optimizer returned parsed JSON or fell back to the original query.
                logger.info(
                    "Optimized queries returned by optimizer: {}", optimized_queries
                )
                timing_breakdown["query_optimization"] = (
                    time.perf_counter() - stage_start
                )
                logger.info(
                    "Query optimization completed in {:.3f}s",
                    timing_breakdown["query_optimization"],
                )
            except Exception as e:
                logger.exception("Query optimization failed for query: {}", query[:100])
                raise QueryOptimizationError(
                    f"Failed to optimize query: {str(e)}"
                ) from e

            # Stage 2: Semantic Scholar Search
            logger.info(
                "Stage 2: Semantic Scholar search - queries: {}",
                [
                    q[:50]
                    for q in [
                        optimized_queries["primary"],
                        optimized_queries["broad"],
                        optimized_queries["alternative"],
                    ]
                ],
            )
            stage_start = time.perf_counter()
            try:
                queries = [
                    optimized_queries["primary"],
                    optimized_queries["broad"],
                    optimized_queries["alternative"],
                ]
                papers = await self._get_client().multi_query_search(
                    queries, limit_per_query=100
                )
                timing_breakdown["semantic_scholar_search"] = (
                    time.perf_counter() - stage_start
                )
                logger.info(
                    "Semantic Scholar search completed in {:.3f}s - found {} papers",
                    timing_breakdown["semantic_scholar_search"],
                    len(papers),
                )
            except Exception as e:
                logger.exception(
                    "Semantic Scholar search failed for queries: {}",
                    [q[:50] for q in queries],
                )
                if "Semantic Scholar" in str(e) or "503" in str(e):
                    raise SemanticScholarError(
                        "Semantic Scholar API is currently unavailable. Please try again later."
                    ) from e
                raise SemanticScholarError(f"Failed to search papers: {str(e)}") from e

            # Stage 3: Quality Gate
            logger.info(
                "Stage 3: Quality gate validation - {} papers to validate", len(papers)
            )
            stage_start = time.perf_counter()
            try:
                # Week 1: No query embedding needed - quality gate uses paper-level metrics only
                # (citations, recency, venue quality)
                query_embedding = None
                quality_result = validate_quality(papers, query_embedding)
                timing_breakdown["quality_gate"] = time.perf_counter() - stage_start
                logger.info(
                    "Quality gate completed in {:.3f}s - passed: {}",
                    timing_breakdown["quality_gate"],
                    quality_result["passed"],
                )
            except Exception as e:
                logger.exception(
                    "Quality gate validation failed for {} papers", len(papers)
                )
                raise QualityGateError(
                    f"Failed to validate paper quality: {str(e)}"
                ) from e

            finalists = []
            if quality_result["passed"]:
                # Stage 4: Composite Scoring
                logger.info(
                    "Stage 4: Composite scoring - ranking {} papers", len(papers)
                )
                stage_start = time.perf_counter()
                try:
                    finalists = self._get_ranker().rank_papers(
                        papers, optimized_queries["final_rephrase"]
                    )
                    timing_breakdown["composite_scoring"] = (
                        time.perf_counter() - stage_start
                    )
                    logger.info(
                        "Composite scoring completed in {:.3f}s - selected {} finalists",
                        timing_breakdown["composite_scoring"],
                        len(finalists),
                    )
                except Exception as e:
                    logger.exception(
                        "Composite scoring failed for {} papers", len(papers)
                    )
                    raise RankingError(f"Failed to rank papers: {str(e)}") from e
            else:
                timing_breakdown["composite_scoring"] = 0.0
                logger.info("Quality gate failed - skipping ranking stage")

            # Success
            execution_time = time.perf_counter() - start_time
            result = PipelineResult(
                query=query,
                optimized_queries=optimized_queries,
                quality_gate=quality_result,
                finalists=finalists,
                execution_time=execution_time,
                timing_breakdown=timing_breakdown,
                success=True,
            )

            logger.info(
                f"Query pipeline completed successfully in {execution_time:.3f}s - quality_gate: {quality_result['passed']}, finalists: {len(finalists)}"
            )
            return result

        except QueryServiceError:
            # Re-raise service-specific errors
            raise
        except Exception as e:
            # Catch any unexpected errors
            execution_time = time.perf_counter() - start_time
            logger.exception(
                "Unexpected error in query pipeline for query: {}", query[:100]
            )
            raise QueryServiceError(
                f"An unexpected error occurred: {str(e)}", "unknown", 500
            ) from e
