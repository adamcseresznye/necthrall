"""Discovery service for stages 1-4 of the pipeline.

Responsibilities:
1. Query Optimization
2. Semantic Scholar Search
3. Quality Gate
4. Composite Scoring
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

import config
from agents.query_optimization_agent import QueryOptimizationAgent
from agents.semantic_scholar_client import SemanticScholarClient
from agents.quality_gate import validate_quality
from agents.ranking_agent import RankingAgent
from services.exceptions import (
    QueryOptimizationError,
    SemanticScholarError,
    QualityGateError,
    RankingError,
)


@dataclass
class DiscoveryResult:
    """Result of the discovery phase."""

    optimized_queries: Dict[str, Any]
    quality_gate: Dict[str, Any]
    finalists: List[Dict[str, Any]]
    timing_breakdown: Dict[str, float]
    refinement_count: int


class DiscoveryService:
    """Service for paper discovery and ranking."""

    def __init__(self):
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

    async def _execute_search_and_quality_gate(
        self,
        queries: List[str],
        timing_breakdown: Dict[str, float],
        attempt_label: str = "",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute Semantic Scholar search and Quality Gate validation.

        Args:
            queries: List of query strings for multi_query_search.
            timing_breakdown: Dict to update with timing info.
            attempt_label: Label for logging (e.g., "" or "_refinement").

        Returns:
            Tuple of (papers, quality_result).

        Raises:
            SemanticScholarError: If search fails.
            QualityGateError: If quality validation fails.
        """
        # Semantic Scholar Search
        logger.info(
            "Stage 2{}: Semantic Scholar search - queries: {}",
            attempt_label,
            [q[:50] for q in queries],
        )
        stage_start = time.perf_counter()
        try:
            papers = await self._get_client().multi_query_search(
                queries, limit_per_query=100
            )
            search_key = f"semantic_scholar_search{attempt_label}"
            timing_breakdown[search_key] = time.perf_counter() - stage_start
            logger.info(
                "Semantic Scholar search{} completed in {:.3f}s - found {} papers",
                attempt_label,
                timing_breakdown[search_key],
                len(papers),
            )
        except Exception as e:
            logger.exception(
                "Semantic Scholar search{} failed for queries: {}",
                attempt_label,
                [q[:50] for q in queries],
            )
            if "Semantic Scholar" in str(e) or "503" in str(e):
                raise SemanticScholarError(
                    "Semantic Scholar API is currently unavailable. Please try again later."
                ) from e
            raise SemanticScholarError(f"Failed to search papers: {str(e)}") from e

        # Quality Gate
        logger.info(
            "Stage 3{}: Quality gate validation - {} papers to validate",
            attempt_label,
            len(papers),
        )
        stage_start = time.perf_counter()
        try:
            query_embedding = None
            quality_result = validate_quality(papers, query_embedding)
            gate_key = f"quality_gate{attempt_label}"
            timing_breakdown[gate_key] = time.perf_counter() - stage_start
            logger.info(
                "Quality gate{} completed in {:.3f}s - passed: {}",
                attempt_label,
                timing_breakdown[gate_key],
                quality_result["passed"],
            )
        except Exception as e:
            logger.exception(
                "Quality gate{} validation failed for {} papers",
                attempt_label,
                len(papers),
            )
            raise QualityGateError(f"Failed to validate paper quality: {str(e)}") from e

        return papers, quality_result

    async def discover(
        self, query: str, optimized_data: Optional[Dict[str, Any]] = None
    ) -> DiscoveryResult:
        """Execute the discovery phase (Stages 1-4).

        Args:
            query: The user query string.
            optimized_data: Optional pre-computed optimization result.

        Returns:
            DiscoveryResult containing optimized queries, quality results, and finalists.
        """
        timing_breakdown = {}
        refinement_count = 0

        # Stage 1: Query Optimization (or use provided data)
        logger.info("Stage 1: Query optimization - input query: {}", query[:100])
        stage_start = time.perf_counter()

        if optimized_data:
            logger.info("Using pre-computed optimization data")
            optimized_queries = optimized_data
            timing_breakdown["query_optimization"] = 0.0
        else:
            try:
                optimized_queries = await self._get_optimizer().generate_dual_queries(
                    query
                )
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

        # Extract search queries based on strategy
        strategy = optimized_queries.get("strategy", "expansion")
        if strategy == "decomposition":
            queries = optimized_queries.get("sub_queries", [])
            # Ensure it's a flat list of strings
            if not isinstance(queries, list):
                queries = [str(queries)]
        else:
            # Expansion strategy
            queries = [
                optimized_queries.get("primary", query),
                optimized_queries.get("broad", query),
                optimized_queries.get("alternative", query),
            ]
        papers, quality_result = await self._execute_search_and_quality_gate(
            queries, timing_breakdown, attempt_label=""
        )

        # Refinement Loop: If quality gate fails on first attempt, try with broad query
        if not quality_result["passed"] and refinement_count == 0:
            logger.warning(
                "⚠️ Quality gate failed on first attempt, attempting refinement with broad query..."
            )
            refinement_count = 1

            # Use broad query as fallback for refinement
            fallback_query = optimized_queries.get("broad", query)
            logger.info(
                "🔄 Refinement attempt {} - using fallback query: {}",
                refinement_count,
                fallback_query[:100],
            )

            # Re-run search and quality gate with fallback query
            refinement_queries = [
                fallback_query,
                optimized_queries.get("alternative", query),
                query,
            ]
            papers, quality_result = await self._execute_search_and_quality_gate(
                refinement_queries, timing_breakdown, attempt_label="_refinement"
            )

            if quality_result["passed"]:
                logger.info(
                    "✅ Refinement successful - quality gate passed on attempt {}",
                    refinement_count + 1,
                )
            else:
                logger.warning(
                    "❌ Refinement failed - quality gate still not passed after {} attempt(s)",
                    refinement_count + 1,
                )

        finalists = []
        if quality_result["passed"]:
            # Stage 4: Composite Scoring
            logger.info("Stage 4: Composite scoring - ranking {} papers", len(papers))
            stage_start = time.perf_counter()
            try:
                finalists = await asyncio.to_thread(
                    self._get_ranker().rank_papers,
                    papers,
                    optimized_queries["final_rephrase"],
                    50,  # top_k for Base+Bonus strategy
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
                logger.exception("Composite scoring failed for {} papers", len(papers))
                raise RankingError(f"Failed to rank papers: {str(e)}") from e
        else:
            timing_breakdown["composite_scoring"] = 0.0
            logger.info("Quality gate failed - skipping ranking stage")

        return DiscoveryResult(
            optimized_queries=optimized_queries,
            quality_gate=quality_result,
            finalists=finalists,
            timing_breakdown=timing_breakdown,
            refinement_count=refinement_count,
        )
