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
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from loguru import logger


from agents.query_optimization_agent import QueryOptimizationAgent
from agents.semantic_scholar_client import SemanticScholarClient
from agents.quality_gate import validate_quality
from agents.ranking_agent import RankingAgent
import config

# These are imported inside methods to prevent torch DLL issues
if TYPE_CHECKING:
    from agents.acquisition_agent import AcquisitionAgent
    from agents.processing_agent import ProcessingAgent
    from agents.synthesis_agent import SynthesisAgent
    from retrieval.llamaindex_retriever import LlamaIndexRetriever
    from retrieval.reranker import CrossEncoderReranker
    from models.state import State
    from utils.citation_verifier import CitationVerifier


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
    optimized_queries: Dict[str, str]
    quality_gate: Dict[str, Any]
    finalists: List[Dict[str, Any]]
    execution_time: float
    timing_breakdown: Dict[str, float]
    success: bool
    error_message: Optional[str] = None
    error_stage: Optional[str] = None
    passages: List[Any] = field(default_factory=list)
    answer: Optional[str] = None
    citation_verification: Optional[Dict[str, Any]] = None
    refinement_count: int = 0


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


class AcquisitionError(QueryServiceError):
    """Error during PDF acquisition."""

    def __init__(self, message: str):
        super().__init__(message, "pdf_acquisition", 500)


class ProcessingError(QueryServiceError):
    """Error during processing and embedding."""

    def __init__(self, message: str):
        super().__init__(message, "processing", 500)


class RetrievalError(QueryServiceError):
    """Error during hybrid retrieval."""

    def __init__(self, message: str):
        super().__init__(message, "retrieval", 500)


class RerankingError(QueryServiceError):
    """Error during cross-encoder reranking."""

    def __init__(self, message: str):
        super().__init__(message, "reranking", 500)


class SynthesisError(QueryServiceError):
    """Error during answer synthesis."""

    def __init__(self, message: str):
        super().__init__(message, "synthesis", 500)


class VerificationError(QueryServiceError):
    """Error during citation verification."""

    def __init__(self, message: str):
        super().__init__(message, "verification", 500)


class QueryService:
    """Service for orchestrating the query pipeline.
    Provides comprehensive error handling and performance monitoring.
    """

    def __init__(self, embedding_model: Any = None):
        """Initialize the query service.

        Args:
            embedding_model: Pre-loaded embedding model for query/passage embedding.
        """
        self.embedding_model = embedding_model
        self._optimizer = None
        self._client = None
        self._ranker = None
        self._acquisition_agent = None
        self._processing_agent = None
        self._retriever = None
        self._synthesis_agent = None
        self._verifier = None

        # Pre-load CrossEncoderReranker during initialization to avoid ~12s latency
        # on first request. Import here to maintain safe DLL import order on Windows.
        from retrieval.reranker import CrossEncoderReranker

        self._reranker = CrossEncoderReranker()

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

    def _get_acquisition_agent(self) -> "AcquisitionAgent":
        """Lazy initialization of PDF acquisition agent."""
        if self._acquisition_agent is None:
            from agents.acquisition_agent import AcquisitionAgent

            self._acquisition_agent = AcquisitionAgent()
        return self._acquisition_agent

    def _get_processing_agent(self) -> "ProcessingAgent":
        """Lazy initialization of processing agent."""
        if self._processing_agent is None:
            from agents.processing_agent import ProcessingAgent

            self._processing_agent = ProcessingAgent(
                chunk_size=512,
                chunk_overlap=50,
            )
        return self._processing_agent

    def _get_retriever(self) -> Optional["LlamaIndexRetriever"]:
        """Lazy initialization of hybrid retriever.

        Returns None if embedding_model is not available.
        """
        if self._retriever is None and self.embedding_model is not None:
            from retrieval.llamaindex_retriever import LlamaIndexRetriever

            self._retriever = LlamaIndexRetriever(
                embedding_model=self.embedding_model,
                top_k=15,  # Before reranking
            )
        return self._retriever

    def _get_reranker(self) -> "CrossEncoderReranker":
        """Return the pre-loaded cross-encoder reranker."""
        return self._reranker

    def _get_synthesis_agent(self) -> "SynthesisAgent":
        """Lazy initialization of synthesis agent."""
        if self._synthesis_agent is None:
            from agents.synthesis_agent import SynthesisAgent

            self._synthesis_agent = SynthesisAgent()
        return self._synthesis_agent

    def _get_verifier(self) -> "CitationVerifier":
        """Lazy initialization of citation verifier."""
        if self._verifier is None:
            from utils.citation_verifier import CitationVerifier

            self._verifier = CitationVerifier()
        return self._verifier

    async def _execute_search_and_quality_gate(
        self,
        queries: List[str],
        timing_breakdown: Dict[str, float],
        attempt_label: str = "",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute Semantic Scholar search and Quality Gate validation.

        Helper method to keep the refinement loop logic flat.

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

    async def process_query(
        self, query: str, progress_callback: Optional[Callable] = None
    ) -> PipelineResult:
        """Process a user query through the complete pipeline.

        Stages:
            1. Query Optimization - Generate optimized query variants
            2. Semantic Scholar Search - Retrieve papers from API
            3. Quality Gate - Validate paper quality
            4. Composite Scoring - Rank papers by relevance
            5. PDF Acquisition - Download PDFs for top papers
            6. Processing & Embedding - Chunk and embed text
            7. Hybrid Retrieval - Vector + BM25 search with RRF fusion
            8. Cross-Encoder Reranking - Final passage ranking
            9. Synthesis - Generate cited answer from passages
            10. Verification - Validate citation references

        Args:
            query: The user query string
            progress_callback: Optional async callback to report progress

        Returns:
            PipelineResult with all processing results and metadata

        Raises:
            QueryServiceError: For pipeline-specific errors
        """
        start_time = time.perf_counter()
        timing_breakdown = {}
        refinement_count = 0
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

            # Stage 2 & 3: Semantic Scholar Search + Quality Gate (with refinement loop)
            await report_progress()
            queries = [
                optimized_queries["primary"],
                optimized_queries["broad"],
                optimized_queries["alternative"],
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
                logger.info(
                    "Stage 4: Composite scoring - ranking {} papers", len(papers)
                )
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
                    logger.exception(
                        "Composite scoring failed for {} papers", len(papers)
                    )
                    raise RankingError(f"Failed to rank papers: {str(e)}") from e
            else:
                timing_breakdown["composite_scoring"] = 0.0
                logger.info("Quality gate failed - skipping ranking stage")

            # =====================================================================
            # PDF Acquisition, Processing, Retrieval, Reranking
            # =====================================================================

            # Check if we have finalists to process
            if not finalists:
                logger.info("ℹ️ No finalists to process")
                execution_time = time.perf_counter() - start_time
                result = PipelineResult(
                    query=query,
                    optimized_queries=optimized_queries,
                    quality_gate=quality_result,
                    finalists=finalists,
                    execution_time=execution_time,
                    timing_breakdown=timing_breakdown,
                    success=True,
                    passages=[],
                    refinement_count=refinement_count,
                )
                return result

            # Check if embedding model is available for stages (5-8)
            if self.embedding_model is None:
                logger.warning(
                    "⚠️ Embedding model not available - skipping stages (5-8). "
                )
                execution_time = time.perf_counter() - start_time
                result = PipelineResult(
                    query=query,
                    optimized_queries=optimized_queries,
                    quality_gate=quality_result,
                    finalists=finalists,
                    execution_time=execution_time,
                    timing_breakdown=timing_breakdown,
                    success=True,
                    passages=[],
                    refinement_count=refinement_count,
                )
                return result

            # Stage 5: PDF Acquisition
            await report_progress()
            logger.info(
                "🧮 Stage 5: PDF Acquisition - downloading PDFs for {} finalists",
                len(finalists),
            )
            stage_start = time.perf_counter()
            passages = []
            state = None
            try:
                from models.state import State

                state = State(query=query, finalists=finalists)
                acquisition_agent = self._get_acquisition_agent()
                state = await acquisition_agent.process(state)
                passages = state.passages or []
                timing_breakdown["pdf_acquisition"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ PDF acquisition completed in {:.3f}s - acquired {} PDFs",
                    timing_breakdown["pdf_acquisition"],
                    len(passages),
                )
            except Exception as e:
                timing_breakdown["pdf_acquisition"] = time.perf_counter() - stage_start
                logger.warning(
                    "⚠️ PDF acquisition failed: {}. Continuing with partial results.",
                    str(e),
                )

            # Check if we have passages to process
            if not passages:
                logger.warning("⚠️ No PDFs acquired")
                execution_time = time.perf_counter() - start_time
                result = PipelineResult(
                    query=query,
                    optimized_queries=optimized_queries,
                    quality_gate=quality_result,
                    finalists=finalists,
                    execution_time=execution_time,
                    timing_breakdown=timing_breakdown,
                    success=True,
                    passages=[],
                    refinement_count=refinement_count,
                )
                return result

            # Stage 6: Processing & Embedding
            await report_progress()
            logger.info(
                "📄 Stage 6: Processing & Embedding - chunking {} passages",
                len(passages),
            )
            stage_start = time.perf_counter()
            chunks = []
            try:
                processing_agent = self._get_processing_agent()
                state = await asyncio.to_thread(
                    processing_agent.process,
                    state,
                    embedding_model=self.embedding_model,
                    batch_size=32,
                )
                chunks = state.chunks or []
                timing_breakdown["processing"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ Processing completed in {:.3f}s - generated {} chunks",
                    timing_breakdown["processing"],
                    len(chunks),
                )
            except Exception as e:
                timing_breakdown["processing"] = time.perf_counter() - stage_start
                logger.warning(
                    "⚠️ Processing failed: {}. Continuing with partial results.",
                    str(e),
                )

            # Check if we have chunks to retrieve from
            if not chunks:
                logger.warning("⚠️ No chunks generated")
                execution_time = time.perf_counter() - start_time
                result = PipelineResult(
                    query=query,
                    optimized_queries=optimized_queries,
                    quality_gate=quality_result,
                    finalists=finalists,
                    execution_time=execution_time,
                    timing_breakdown=timing_breakdown,
                    success=True,
                    passages=[],
                    refinement_count=refinement_count,
                )
                return result

            # Stage 7: Hybrid Retrieval
            logger.info(
                "🔍 Stage 7: Hybrid Retrieval - searching {} chunks",
                len(chunks),
            )
            stage_start = time.perf_counter()
            retrieval_results = []
            try:
                retriever = self._get_retriever()
                if retriever is None:
                    raise RetrievalError(
                        "Retriever not available (embedding model missing)"
                    )

                # Use final_rephrase for retrieval query
                retrieval_query = optimized_queries.get("final_rephrase", query)

                # Extract pre-computed embeddings from chunks (computed in Stage 6)
                chunk_embeddings = [
                    chunk.metadata["embedding"]
                    for chunk in chunks
                    if "embedding" in chunk.metadata
                ]

                # Use pre-computed embeddings path if all embeddings are available
                if len(chunk_embeddings) == len(chunks):
                    logger.info("🚀 Using pre-computed embeddings for retrieval...")
                    # Wrap in asyncio.to_thread
                    retrieval_results = await asyncio.to_thread(
                        retriever.retrieve_with_embeddings,
                        query=retrieval_query,
                        chunks=chunks,
                        chunk_embeddings=chunk_embeddings,
                    )
                else:
                    logger.warning("⚠️ Missing embeddings, falling back...")
                    # Wrap in asyncio.to_thread
                    retrieval_results = await asyncio.to_thread(
                        retriever.retrieve,
                        query=retrieval_query,
                        chunks=chunks,
                    )

                timing_breakdown["retrieval"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ Hybrid retrieval completed in {:.3f}s - found {} passages",
                    timing_breakdown["retrieval"],
                    len(retrieval_results),
                )
            except Exception as e:
                timing_breakdown["retrieval"] = time.perf_counter() - stage_start
                logger.warning(
                    "⚠️ Retrieval failed: {}. Continuing with partial results.",
                    str(e),
                )

            # Check if we have retrieval results to rerank
            if not retrieval_results:
                logger.warning("⚠️ No retrieval results")
                execution_time = time.perf_counter() - start_time
                result = PipelineResult(
                    query=query,
                    optimized_queries=optimized_queries,
                    quality_gate=quality_result,
                    finalists=finalists,
                    execution_time=execution_time,
                    timing_breakdown=timing_breakdown,
                    success=True,
                    passages=[],
                    refinement_count=refinement_count,
                )
                return result

            # Stage 8: Cross-Encoder Reranking
            logger.info(
                "🎯 Stage 8: Cross-Encoder Reranking - reranking {} passages",
                len(retrieval_results),
            )
            stage_start = time.perf_counter()
            final_passages = []
            try:
                reranker = self._get_reranker()
                # Use final_rephrase for reranking query
                rerank_query = optimized_queries.get("final_rephrase", query)
                final_passages = await asyncio.to_thread(
                    reranker.rerank,
                    query=rerank_query,
                    nodes=retrieval_results,
                    top_k=12,
                )
                timing_breakdown["reranking"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ Reranking completed in {:.3f}s - selected {} final passages",
                    timing_breakdown["reranking"],
                    len(final_passages),
                )
            except Exception as e:
                timing_breakdown["reranking"] = time.perf_counter() - stage_start
                logger.warning(
                    "⚠️ Reranking failed: {}. Returning retrieval results as-is.",
                    str(e),
                )
                # Fallback: use retrieval results without reranking
                final_passages = retrieval_results[:12]

            # =====================================================================
            # Stages 9-10: Synthesis and Verification
            # =====================================================================

            answer = None
            citation_verification = None

            # Stage 9: Synthesis
            if final_passages:
                await report_progress()
                logger.info(
                    "📝 Stage 9: Synthesis - generating answer from {} passages",
                    len(final_passages),
                )
                stage_start = time.perf_counter()
                try:
                    synthesis_agent = self._get_synthesis_agent()
                    answer = await synthesis_agent.synthesize(query, final_passages)
                    timing_breakdown["synthesis"] = time.perf_counter() - stage_start
                    logger.info(
                        "✅ Synthesis completed in {:.3f}s - answer length: {} chars",
                        timing_breakdown["synthesis"],
                        len(answer) if answer else 0,
                    )
                except Exception as e:
                    timing_breakdown["synthesis"] = time.perf_counter() - stage_start
                    logger.warning(
                        "⚠️ Synthesis failed: {}. Continuing without answer.",
                        str(e),
                    )

                # Stage 10: Verification
                if answer:
                    logger.info(
                        "🔍 Stage 10: Verification - validating citations in answer"
                    )
                    stage_start = time.perf_counter()
                    try:
                        verifier = self._get_verifier()
                        citation_verification = await asyncio.to_thread(
                            verifier.verify, answer, final_passages
                        )
                        timing_breakdown["verification"] = (
                            time.perf_counter() - stage_start
                        )
                        logger.info(
                            "✅ Verification completed in {:.3f}s - valid: {}",
                            timing_breakdown["verification"],
                            citation_verification.get("valid", False),
                        )
                    except Exception as e:
                        timing_breakdown["verification"] = (
                            time.perf_counter() - stage_start
                        )
                        logger.warning(
                            "⚠️ Verification failed: {}. Continuing without verification.",
                            str(e),
                        )

            execution_time = time.perf_counter() - start_time
            result = PipelineResult(
                query=query,
                optimized_queries=optimized_queries,
                quality_gate=quality_result,
                finalists=finalists,
                execution_time=execution_time,
                timing_breakdown=timing_breakdown,
                success=True,
                passages=final_passages,
                answer=answer,
                citation_verification=citation_verification,
                refinement_count=refinement_count,
            )

            logger.info(
                "✅ Full pipeline completed in {:.3f}s - "
                "quality_gate: {}, finalists: {}, passages: {}, answer: {}, refinements: {}",
                execution_time,
                quality_result["passed"],
                len(finalists),
                len(final_passages),
                "yes" if answer else "no",
                refinement_count,
            )
            return result

        except QueryServiceError:
            # Re-raise service-specific errors
            raise
        except Exception as e:
            # Catch any unexpected errors
            execution_time = time.perf_counter() - start_time
            logger.exception(
                "❌ Unexpected error in query pipeline for query: {}", query[:100]
            )
            raise QueryServiceError(
                f"An unexpected error occurred: {str(e)}", "unknown", 500
            ) from e
