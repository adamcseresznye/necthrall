from typing import Dict, List, Any, Tuple
from datetime import datetime
import json
import asyncio
import time
from loguru import logger
import re

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError
from google.api_core.exceptions import ResourceExhausted, InvalidArgument
import httpx

from models.state import (
    State,
    CredibilityScore,
    DetectedContradiction,
    ContradictionClaim,
)
from prompts.analysis_prompts import CONTRADICTION_DETECTION_PROMPT
from monitoring.health import HealthMonitor
from utils.error_handling import (
    LLMRetryConfig,
    tenacity_retry_decorator,
    is_retryable_exception,
)
from dotenv import load_dotenv
import os

load_dotenv()


class AnalysisError(Exception):
    """Base exception for analysis-related errors."""

    def __init__(
        self,
        message: str,
        component: str,
        recoverable: bool = True,
        context: Dict[str, Any] = None,
    ):
        super().__init__(message)
        self.component = component
        self.recoverable = recoverable
        self.context = context or {}


class CredibilityScoringError(AnalysisError):
    """Error during credibility scoring operations."""

    def __init__(
        self, message: str, recoverable: bool = True, context: Dict[str, Any] = None
    ):
        super().__init__(message, "credibility_scorer", recoverable, context)


class ContradictionDetectionError(AnalysisError):
    """Error during contradiction detection operations."""

    def __init__(
        self, message: str, recoverable: bool = True, context: Dict[str, Any] = None
    ):
        super().__init__(message, "contradiction_detector", recoverable, context)


class LLMProviderError(ContradictionDetectionError):
    """Error from LLM provider operations."""

    def __init__(
        self,
        message: str,
        provider: str,
        recoverable: bool = True,
        context: Dict[str, Any] = None,
    ):
        super().__init__(message, recoverable, context)
        self.provider = provider


class DataValidationError(AnalysisError):
    """Error when input data fails validation."""

    def __init__(
        self,
        message: str,
        component: str,
        recoverable: bool = True,
        context: Dict[str, Any] = None,
    ):
        super().__init__(message, component, recoverable, context)


class RecoveryStrategy:
    """Defines recovery strategies for different error types."""

    @staticmethod
    def for_credibility_error(error: CredibilityScoringError) -> Dict[str, Any]:
        """Recovery strategy for credibility scoring errors."""
        if not error.recoverable:
            return {
                "action": "fail_fast",
                "reason": "Non-recoverable credibility scoring error",
                "fallback": "skip_credibility_scoring",
            }

        # For recoverable errors, try partial scoring
        return {
            "action": "partial_recovery",
            "reason": "Attempt partial credibility scoring for remaining papers",
            "fallback": "default_scores",
            "default_score": 50,  # Medium credibility
            "default_tier": "medium",
        }

    @staticmethod
    def for_contradiction_error(error: ContradictionDetectionError) -> Dict[str, Any]:
        """Recovery strategy for contradiction detection errors."""
        if isinstance(error, LLMProviderError):
            if error.provider == "gemini":
                return {
                    "action": "fallback_provider",
                    "reason": "Switch to Groq provider",
                    "fallback": "groq_provider",
                }
            elif error.provider == "groq":
                return {
                    "action": "reduced_scope",
                    "reason": "Reduce passage count and retry",
                    "fallback": "fewer_passages",
                    "max_passages": 5,
                }

        if not error.recoverable:
            return {
                "action": "fail_fast",
                "reason": "Non-recoverable contradiction detection error",
                "fallback": "skip_contradiction_detection",
            }

        # For recoverable errors, try with reduced scope
        return {
            "action": "reduced_scope",
            "reason": "Attempt contradiction detection with reduced scope",
            "fallback": "fewer_passages",
            "max_passages": 7,
        }

    @staticmethod
    def for_data_validation_error(error: DataValidationError) -> Dict[str, Any]:
        """Recovery strategy for data validation errors."""
        return {
            "action": "data_repair",
            "reason": "Attempt to repair or filter invalid data",
            "fallback": "filter_invalid_data",
        }


class AnalysisAgent:
    """
    Analysis Agent that performs credibility scoring and contradiction detection.

    Combines AnalysisCredibilityScorer and ContradictionDetector to analyze
    filtered papers and relevant passages, returning credibility scores and
    detected contradictions.
    """

    def __init__(self):
        self.credibility_scorer = AnalysisCredibilityScorer()
        self.contradiction_detector = ContradictionDetector()

    async def analyze(self, state: "State") -> Dict[str, Any]:
        """
        Analyze filtered papers and passages for credibility and contradictions.

        Args:
            state: LangGraph State with filtered_papers and relevant_passages

        Returns:
            Dict with credibility_scores, contradictions, and execution_times
        """
        start_time = time.time()

        logger.info(
            json.dumps(
                {
                    "event": "analysis_node_entry",
                    "component": "analysis_agent",
                    "papers_count": len(state.filtered_papers),
                    "passages_count": len(state.relevant_passages),
                    "request_id": getattr(state, "request_id", "unknown"),
                }
            )
        )

        # Initialize results
        credibility_scores = []
        contradictions = []
        analysis_errors = []
        recovery_actions = []

        try:
            # Credibility scoring
            logger.info(
                json.dumps(
                    {
                        "event": "analysis_component_start",
                        "component": "credibility_scorer",
                        "papers_count": len(state.filtered_papers),
                    }
                )
            )

            credibility_start = time.time()

            try:
                credibility_scores, credibility_errors = (
                    await self._perform_credibility_scoring(state)
                )

                credibility_time = time.time() - credibility_start
                logger.info(
                    json.dumps(
                        {
                            "event": "analysis_component_success",
                            "component": "credibility_scorer",
                            "scored_papers": len(credibility_scores),
                            "errors_count": len(credibility_errors),
                            "execution_time": round(credibility_time, 3),
                            "success": True,
                        }
                    )
                )

                # Add individual paper scoring errors
                analysis_errors.extend(credibility_errors)

            except Exception as e:
                credibility_time = time.time() - credibility_start
                error = self._categorize_credibility_error(e, state)
                analysis_errors.append(error)

                recovery = RecoveryStrategy.for_credibility_error(error)
                recovery_actions.append(recovery)

                logger.error(
                    json.dumps(
                        {
                            "event": "analysis_component_error",
                            "component": "credibility_scorer",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "recoverable": error.recoverable,
                            "recovery_action": recovery["action"],
                            "execution_time": round(credibility_time, 3),
                            "success": False,
                        }
                    )
                )

                # Apply recovery strategy
                if recovery["action"] == "partial_recovery":
                    credibility_scores = self._apply_credibility_recovery(
                        state, recovery
                    )
                elif recovery["action"] == "default_scores":
                    credibility_scores = self._generate_default_scores(state, recovery)

        except Exception as e:
            logger.error(
                json.dumps(
                    {
                        "event": "analysis_component_error",
                        "component": "credibility_scorer",
                        "error_type": "UnexpectedError",
                        "error_message": str(e),
                        "recoverable": False,
                        "success": False,
                    }
                )
            )
            analysis_errors.append(
                CredibilityScoringError(f"Unexpected error: {e}", recoverable=False)
            )

        try:
            # Contradiction detection
            logger.info(
                json.dumps(
                    {
                        "event": "analysis_component_start",
                        "component": "contradiction_detector",
                        "passages_count": len(state.relevant_passages),
                    }
                )
            )

            contradiction_start = time.time()

            try:
                if state.relevant_passages:
                    contradictions = await self._perform_contradiction_detection(state)
                else:
                    logger.info(
                        json.dumps(
                            {
                                "event": "analysis_component_skip",
                                "component": "contradiction_detector",
                                "reason": "no_passages_available",
                            }
                        )
                    )

                contradiction_time = time.time() - contradiction_start
                logger.info(
                    json.dumps(
                        {
                            "event": "analysis_component_success",
                            "component": "contradiction_detector",
                            "contradictions_found": len(contradictions),
                            "execution_time": round(contradiction_time, 3),
                            "success": True,
                        }
                    )
                )

            except Exception as e:
                contradiction_time = time.time() - contradiction_start
                error = self._categorize_contradiction_error(e, state)
                analysis_errors.append(error)

                recovery = RecoveryStrategy.for_contradiction_error(error)
                recovery_actions.append(recovery)

                logger.error(
                    json.dumps(
                        {
                            "event": "analysis_component_error",
                            "component": "contradiction_detector",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "recoverable": error.recoverable,
                            "recovery_action": recovery["action"],
                            "execution_time": round(contradiction_time, 3),
                            "success": False,
                        }
                    )
                )

                # Apply recovery strategy
                if recovery["action"] == "reduced_scope":
                    contradictions = await self._apply_contradiction_recovery(
                        state, recovery
                    )
                elif recovery["action"] == "fallback_provider":
                    contradictions = await self._retry_with_fallback_provider(
                        state, recovery
                    )

        except Exception as e:
            logger.error(
                json.dumps(
                    {
                        "event": "analysis_component_error",
                        "component": "contradiction_detector",
                        "error_type": "UnexpectedError",
                        "error_message": str(e),
                        "recoverable": False,
                        "success": False,
                    }
                )
            )
            analysis_errors.append(
                ContradictionDetectionError(f"Unexpected error: {e}", recoverable=False)
            )

        # Calculate total execution time
        total_time = time.time() - start_time

        # Update execution times
        execution_times = dict(state.execution_times)  # Copy existing
        execution_times["analysis_agent"] = total_time
        execution_times["credibility_scoring"] = locals().get("credibility_time", 0.0)
        execution_times["contradiction_detection"] = locals().get(
            "contradiction_time", 0.0
        )

        logger.info(
            json.dumps(
                {
                    "event": "analysis_node_exit",
                    "component": "analysis_agent",
                    "total_execution_time": round(total_time, 3),
                    "credibility_scores": len(credibility_scores),
                    "contradictions": len(contradictions),
                    "errors_count": len(analysis_errors),
                    "recovery_actions": len(recovery_actions),
                    "success": len(analysis_errors) == 0,
                }
            )
        )

        return {
            "credibility_scores": credibility_scores,
            "contradictions": contradictions,
            "execution_times": execution_times,
            "analysis_errors": [str(e) for e in analysis_errors],
            "recovery_actions": recovery_actions,
        }

    def _categorize_credibility_error(
        self, error: Exception, state: State
    ) -> CredibilityScoringError:
        """Categorize credibility scoring errors."""
        error_msg = str(error)

        # If it's already a categorized error, return it
        if isinstance(
            error, (LLMProviderError, DataValidationError, CredibilityScoringError)
        ):
            return error

        if isinstance(error, (ValueError, TypeError)):
            return DataValidationError(
                f"Data validation error in credibility scoring: {error_msg}",
                "credibility_scorer",
                recoverable=True,
                context={"papers_count": len(state.filtered_papers)},
            )
        elif isinstance(error, MemoryError):
            return CredibilityScoringError(
                f"Memory error during credibility scoring: {error_msg}",
                recoverable=False,
                context={"papers_count": len(state.filtered_papers)},
            )
        else:
            return CredibilityScoringError(
                f"Unexpected credibility scoring error: {error_msg}",
                recoverable=True,
                context={"papers_count": len(state.filtered_papers)},
            )

    def _categorize_contradiction_error(
        self, error: Exception, state: State
    ) -> ContradictionDetectionError:
        """Categorize contradiction detection errors."""
        error_msg = str(error)

        # If it's already a categorized error, return it
        if isinstance(
            error, (LLMProviderError, DataValidationError, ContradictionDetectionError)
        ):
            return error

        if isinstance(
            error, (ResourceExhausted, httpx.TimeoutException, asyncio.TimeoutError)
        ):
            return LLMProviderError(
                f"LLM provider timeout/rate limit: {error_msg}",
                provider="unknown",
                recoverable=True,
                context={"passages_count": len(state.relevant_passages)},
            )
        elif isinstance(error, InvalidArgument):
            return LLMProviderError(
                f"LLM provider authentication error: {error_msg}",
                provider="unknown",
                recoverable=False,
                context={"passages_count": len(state.relevant_passages)},
            )
        elif isinstance(error, ValidationError):
            return DataValidationError(
                f"LLM response validation error: {error_msg}",
                "contradiction_detector",
                recoverable=True,
                context={"passages_count": len(state.relevant_passages)},
            )
        else:
            return ContradictionDetectionError(
                f"Unexpected contradiction detection error: {error_msg}",
                recoverable=True,
                context={"passages_count": len(state.relevant_passages)},
            )

    async def _perform_credibility_scoring(
        self, state: State
    ) -> Tuple[List[CredibilityScore], List[str]]:
        """Perform credibility scoring with error handling."""
        if not state.filtered_papers:
            raise DataValidationError(
                "No filtered papers available for scoring", "credibility_scorer"
            )

        scores = []
        errors = []
        for paper in state.filtered_papers:
            try:
                paper_dict = {
                    "paper_id": paper.paper_id,
                    "citation_count": paper.citation_count,
                    "year": paper.year,
                    "journal": paper.journal,
                }
                score = self.credibility_scorer.score_paper(paper_dict)
                scores.append(score)
            except Exception as e:
                error_msg = f"Failed to score paper {paper.paper_id}: {str(e)}"
                logger.warning(
                    json.dumps(
                        {
                            "event": "paper_scoring_failed",
                            "paper_id": paper.paper_id,
                            "error": str(e),
                        }
                    )
                )
                errors.append(error_msg)
                # Add default score for failed papers
                scores.append(
                    CredibilityScore(
                        paper_id=paper.paper_id,
                        score=50,
                        tier="medium",
                        rationale="Default score due to scoring failure",
                    )
                )

        return scores, errors

    async def _perform_contradiction_detection(
        self, state: State
    ) -> List[DetectedContradiction]:
        """Perform contradiction detection with error handling."""
        if not state.relevant_passages:
            return []

        passages_dict = []
        for passage in state.relevant_passages:
            passages_dict.append(
                {
                    "paper_id": passage.paper_id,
                    "text": passage.content,
                    "paper_title": getattr(passage, "paper_title", ""),
                }
            )

        llm_config = state.config.__dict__ if hasattr(state.config, "__dict__") else {}

        return await self.contradiction_detector.detect_contradictions(
            query=state.optimized_query or state.original_query,
            relevant_passages=passages_dict,
            llm_config=llm_config,
        )

    def _apply_credibility_recovery(
        self, state: State, recovery: Dict[str, Any]
    ) -> List[CredibilityScore]:
        """Apply credibility scoring recovery strategy."""
        scores = []
        for paper in state.filtered_papers:
            try:
                paper_dict = {
                    "paper_id": paper.paper_id,
                    "citation_count": paper.citation_count,
                    "year": paper.year,
                    "journal": paper.journal,
                }
                score = self.credibility_scorer.score_paper(paper_dict)
                scores.append(score)
            except Exception:
                # Apply default score for failed papers
                default_score = recovery.get("default_score", 50)
                default_tier = recovery.get("default_tier", "medium")
                scores.append(
                    CredibilityScore(
                        paper_id=paper.paper_id,
                        score=default_score,
                        tier=default_tier,
                        rationale="Default score due to scoring failure",
                    )
                )
        return scores

    def _generate_default_scores(
        self, state: State, recovery: Dict[str, Any]
    ) -> List[CredibilityScore]:
        """Generate default credibility scores for all papers."""
        default_score = recovery.get("default_score", 50)
        default_tier = recovery.get("default_tier", "medium")

        return [
            CredibilityScore(
                paper_id=paper.paper_id,
                score=default_score,
                tier=default_tier,
                rationale="Default score due to complete scoring failure",
            )
            for paper in state.filtered_papers
        ]

    async def _apply_contradiction_recovery(
        self, state: State, recovery: Dict[str, Any]
    ) -> List[DetectedContradiction]:
        """Apply contradiction detection recovery with reduced scope."""
        max_passages = recovery.get("max_passages", 5)

        if len(state.relevant_passages) <= max_passages:
            return await self._perform_contradiction_detection(state)

        # Reduce scope by selecting top passages
        reduced_passages = state.relevant_passages[:max_passages]
        reduced_state = state.model_copy()
        reduced_state.relevant_passages = reduced_passages

        try:
            return await self._perform_contradiction_detection(reduced_state)
        except Exception:
            logger.warning(
                json.dumps(
                    {
                        "event": "recovery_attempt_failed",
                        "component": "contradiction_detector",
                        "recovery_action": "reduced_scope",
                    }
                )
            )
            return []

    async def _retry_with_fallback_provider(
        self, state: State, recovery: Dict[str, Any]
    ) -> List[DetectedContradiction]:
        """Retry contradiction detection with fallback provider."""
        # This would modify the LLM config to use fallback provider
        # For now, just attempt detection again (tenacity will handle provider fallback)
        try:
            return await self._perform_contradiction_detection(state)
        except Exception:
            logger.warning(
                json.dumps(
                    {
                        "event": "recovery_attempt_failed",
                        "component": "contradiction_detector",
                        "recovery_action": "fallback_provider",
                    }
                )
            )
            return []

    # Performance optimization methods
    def _preallocate_credibility_scores(
        self, papers_count: int
    ) -> List[CredibilityScore]:
        """Pre-allocate credibility scores list with estimated capacity to minimize reallocations."""
        # Estimate capacity based on input size, with some buffer for efficiency
        estimated_capacity = max(papers_count + 10, int(papers_count * 1.2))
        return [None] * estimated_capacity  # Pre-allocate with None placeholders

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def _start_performance_monitoring(self, state: State) -> Dict[str, Any]:
        """Initialize performance monitoring data."""
        return {
            "start_time": time.time(),
            "initial_memory": self._memory_baseline or 0.0,
            "papers_count": len(state.filtered_papers),
            "passages_count": len(state.relevant_passages),
        }

    def _finalize_performance_monitoring(
        self,
        state: State,
        component_metrics: Dict[str, Any],
        total_time: float,
        final_memory: float,
    ) -> Dict[str, Any]:
        """Finalize and return performance monitoring data."""
        if not self.performance_monitoring:
            return {}

        memory_delta = final_memory - (self._memory_baseline or 0.0)

        return {
            "total_execution_time": round(total_time, 3),
            "memory_delta_mb": round(memory_delta, 2),
            "peak_memory_mb": max(
                component_metrics["credibility_scoring"]["memory_peak"],
                component_metrics["contradiction_detection"]["memory_peak"],
            ),
            "component_breakdown": {
                "credibility_scoring": {
                    "execution_time": round(
                        component_metrics["credibility_scoring"]["end_time"]
                        - component_metrics["credibility_scoring"]["start_time"],
                        3,
                    ),
                    "memory_peak": component_metrics["credibility_scoring"][
                        "memory_peak"
                    ],
                },
                "contradiction_detection": {
                    "execution_time": round(
                        component_metrics["contradiction_detection"]["end_time"]
                        - component_metrics["contradiction_detection"]["start_time"],
                        3,
                    ),
                    "memory_peak": component_metrics["contradiction_detection"][
                        "memory_peak"
                    ],
                },
            },
            "throughput": {
                "papers_per_second": (
                    round(len(state.filtered_papers) / total_time, 2)
                    if total_time > 0
                    else 0
                ),
                "passages_per_second": (
                    round(len(state.relevant_passages) / total_time, 2)
                    if total_time > 0
                    else 0
                ),
            },
            "efficiency_score": self._calculate_efficiency_score(
                state, total_time, memory_delta
            ),
        }

    def _calculate_efficiency_score(
        self, state: State, total_time: float, memory_delta: float
    ) -> float:
        """Calculate an efficiency score based on performance metrics."""
        # Simple efficiency score: lower is better
        # Factors: time per item, memory per item
        papers_count = len(state.filtered_papers) or 1
        passages_count = len(state.relevant_passages) or 1

        time_efficiency = total_time / (papers_count + passages_count)
        memory_efficiency = memory_delta / (papers_count + passages_count)

        # Normalize and combine (arbitrary weights)
        efficiency_score = (time_efficiency * 0.7) + (memory_efficiency * 0.3)

        return round(efficiency_score, 3)

    def _should_reduce_contradiction_scope(self, state: State) -> bool:
        """Determine if contradiction detection should use reduced scope for performance."""
        passages_count = len(state.relevant_passages)

        # Reduce scope for large passage sets to maintain performance
        if passages_count > 50:
            return True

        # Check memory usage trend
        current_memory = self._get_memory_usage()
        if (
            self._memory_baseline and (current_memory - self._memory_baseline) > 100
        ):  # Over 100MB increase
            return True

        return False

    async def _perform_credibility_scoring_optimized(
        self, state: State, preallocated_scores: List[CredibilityScore]
    ) -> Tuple[List[CredibilityScore], List[str]]:
        """Perform credibility scoring with memory-efficient processing."""
        if not state.filtered_papers:
            raise DataValidationError(
                "No filtered papers available for scoring", "credibility_scorer"
            )

        scores = []
        errors = []
        batch_size = min(
            50, len(state.filtered_papers)
        )  # Process in batches to control memory

        for i in range(0, len(state.filtered_papers), batch_size):
            batch = state.filtered_papers[i : i + batch_size]

            for paper in batch:
                try:
                    paper_dict = {
                        "paper_id": paper.paper_id,
                        "citation_count": paper.citation_count,
                        "year": paper.year,
                        "journal": paper.journal,
                    }
                    score = self.credibility_scorer.score_paper(paper_dict)
                    scores.append(score)
                except Exception as e:
                    error_msg = f"Failed to score paper {paper.paper_id}: {str(e)}"
                    logger.warning(
                        json.dumps(
                            {
                                "event": "paper_scoring_failed",
                                "paper_id": paper.paper_id,
                                "error": str(e),
                            }
                        )
                    )
                    errors.append(error_msg)
                    # Add default score for failed papers
                    scores.append(
                        CredibilityScore(
                            paper_id=paper.paper_id,
                            score=50,
                            tier="medium",
                            rationale="Default score due to scoring failure",
                        )
                    )

            # Yield control to event loop periodically for large batches
            if len(state.filtered_papers) > 20:
                await asyncio.sleep(0)  # Allow other coroutines to run

        return scores, errors

    async def _perform_contradiction_detection_optimized(
        self, state: State
    ) -> List[DetectedContradiction]:
        """Perform contradiction detection with memory-efficient processing."""
        if not state.relevant_passages:
            return []

        # For large passage sets, process in optimized chunks
        if len(state.relevant_passages) > 20:
            return await self._perform_contradiction_detection_batched(state)
        else:
            return await self._perform_contradiction_detection(state)

    async def _perform_contradiction_detection_batched(
        self, state: State
    ) -> List[DetectedContradiction]:
        """Perform batched contradiction detection for large passage sets."""
        passages = state.relevant_passages
        batch_size = 15  # Optimal batch size for LLM context limits
        all_contradictions = []

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]

            # Create temporary state with batch
            batch_state = state.model_copy()
            batch_state.relevant_passages = batch_passages

            try:
                batch_contradictions = await self._perform_contradiction_detection(
                    batch_state
                )
                all_contradictions.extend(batch_contradictions)

                # Yield control between batches
                await asyncio.sleep(0)

            except Exception as e:
                logger.warning(
                    json.dumps(
                        {
                            "event": "batch_contradiction_failed",
                            "batch_start": i,
                            "batch_size": len(batch_passages),
                            "error": str(e),
                        }
                    )
                )
                # Continue with next batch

        # Remove duplicates that might occur across batches
        seen_topics = set()
        unique_contradictions = []
        for contradiction in all_contradictions:
            topic_key = (
                contradiction.topic,
                contradiction.claim_1.text,
                contradiction.claim_2.text,
            )
            if topic_key not in seen_topics:
                seen_topics.add(topic_key)
                unique_contradictions.append(contradiction)

        return unique_contradictions[:3]  # Limit to top 3 overall

    async def _perform_contradiction_detection_reduced(
        self, state: State
    ) -> List[DetectedContradiction]:
        """Perform contradiction detection with reduced scope for performance."""
        # Select top passages by relevance score for reduced processing
        passages = state.relevant_passages
        if len(passages) <= 10:
            return await self._perform_contradiction_detection(state)

        # Sort by retrieval score and take top 10
        sorted_passages = sorted(
            passages,
            key=lambda p: getattr(
                p, "retrieval_score", getattr(p.scores, "final_score", 0.0)
            ),
            reverse=True,
        )[:10]

        # Create reduced state
        reduced_state = state.model_copy()
        reduced_state.relevant_passages = sorted_passages

        return await self._perform_contradiction_detection(reduced_state)

    def _apply_credibility_recovery_optimized(
        self,
        state: State,
        recovery: Dict[str, Any],
        preallocated_scores: List[CredibilityScore],
    ) -> List[CredibilityScore]:
        """Apply credibility scoring recovery with memory optimization."""
        scores = []
        default_score = recovery.get("default_score", 50)
        default_tier = recovery.get("default_tier", "medium")

        for paper in state.filtered_papers:
            try:
                paper_dict = {
                    "paper_id": paper.paper_id,
                    "citation_count": paper.citation_count,
                    "year": paper.year,
                    "journal": paper.journal,
                }
                score = self.credibility_scorer.score_paper(paper_dict)
                scores.append(score)
            except Exception:
                # Apply default score for failed papers
                scores.append(
                    CredibilityScore(
                        paper_id=paper.paper_id,
                        score=default_score,
                        tier=default_tier,
                        rationale="Default score due to scoring failure",
                    )
                )
        return scores

    def _generate_default_scores_optimized(
        self,
        state: State,
        recovery: Dict[str, Any],
        preallocated_scores: List[CredibilityScore],
    ) -> List[CredibilityScore]:
        """Generate default credibility scores with memory optimization."""
        default_score = recovery.get("default_score", 50)
        default_tier = recovery.get("default_tier", "medium")

        return [
            CredibilityScore(
                paper_id=paper.paper_id,
                score=default_score,
                tier=default_tier,
                rationale="Default score due to complete scoring failure",
            )
            for paper in state.filtered_papers
        ]

    async def _apply_contradiction_recovery_optimized(
        self, state: State, recovery: Dict[str, Any]
    ) -> List[DetectedContradiction]:
        """Apply contradiction detection recovery with performance optimization."""
        max_passages = recovery.get("max_passages", 5)

        if len(state.relevant_passages) <= max_passages:
            return await self._perform_contradiction_detection_optimized(state)

        # Reduce scope by selecting top passages
        sorted_passages = sorted(
            state.relevant_passages,
            key=lambda p: getattr(
                p, "retrieval_score", getattr(p.scores, "final_score", 0.0)
            ),
            reverse=True,
        )[:max_passages]

        reduced_state = state.model_copy()
        reduced_state.relevant_passages = sorted_passages

        try:
            return await self._perform_contradiction_detection_optimized(reduced_state)
        except Exception:
            logger.warning(
                json.dumps(
                    {
                        "event": "recovery_attempt_failed",
                        "component": "contradiction_detector",
                        "recovery_action": "reduced_scope",
                    }
                )
            )
            return []

    async def _retry_with_fallback_provider_optimized(
        self, state: State, recovery: Dict[str, Any]
    ) -> List[DetectedContradiction]:
        """Retry contradiction detection with fallback provider and performance optimization."""
        try:
            return await self._perform_contradiction_detection_optimized(state)
        except Exception:
            logger.warning(
                json.dumps(
                    {
                        "event": "recovery_attempt_failed",
                        "component": "contradiction_detector",
                        "recovery_action": "fallback_provider",
                    }
                )
            )
            return []


class AnalysisCredibilityScorer:
    """Credibility scorer for scientific papers with robust error handling and lightweight performance tuning.

    Features:
    - Defensive parsing of metadata fields (None, wrong types)
    - Cached current year and precomputed sets for O(1) venue checks
    - Structured DEBUG logs with per-component contributions
    - Small and allocation-light code paths for fast scoring in tight loops
    """

    # Citation tiers
    CITATION_HIGH = 40
    CITATION_MEDIUM = 30
    CITATION_LOW = 10

    # Recency tiers
    RECENT_2022_PLUS = 30
    RECENT_2018_2021 = 20
    RECENT_OLDER = 10

    # Venue scores
    TOP_JOURNAL_LIST = ["nature", "science", "cell", "nejm", "lancet", "jama"]
    PREPRINT_KEYWORDS = ["arxiv", "preprint", "biorxiv", "medrxiv"]
    TOP_JOURNAL_SET = set(TOP_JOURNAL_LIST)
    PREPRINT_SET = set(PREPRINT_KEYWORDS)

    TOP_JOURNAL_SCORE = 30
    PREPRINT_PENALTY = -10

    # Final tier thresholds
    TIER_HIGH = 75
    TIER_MEDIUM = 50

    # Cache current year once at module/class load to avoid repeated datetime calls
    CURRENT_YEAR = datetime.utcnow().year

    @classmethod
    def _clamp_score(cls, score: int) -> int:
        return max(0, min(100, int(round(score))))

    @classmethod
    def _normalize_journal(cls, journal: Any) -> str:
        """Normalize journal input to a lowercase string; handle non-string inputs gracefully."""
        if journal is None:
            return ""
        if not isinstance(journal, str):
            try:
                journal = str(journal)
            except Exception:
                return ""
        return journal.strip().lower()

    @classmethod
    def _safe_int(cls, value: Any, default: int = 0) -> int:
        """Safely parse integers for citation counts and years; handle floats/strings, return default on failure."""
        if value is None:
            return default
        try:
            return int(float(value))
        except Exception:
            return default

    @classmethod
    def score_paper(cls, metadata: Dict[str, Any]) -> CredibilityScore:
        """Score a single paper metadata dict and return a CredibilityScore.

        Error handling:
        - Non-dict metadata -> default medium score
        - Missing fields are handled with safe defaults
        - Future publication years are clamped to CURRENT_YEAR

        Returns CredibilityScore (score 0-100, tier, rationale).
        """

        default_rationale = "insufficient metadata, default medium credibility"

        # Fast path validation
        if not isinstance(metadata, dict):
            logger.debug(
                json.dumps({"event": "credibility_error", "error": "metadata_not_dict"})
            )
            return CredibilityScore(
                paper_id="", score=50, tier="medium", rationale=default_rationale
            )

        # If no useful metadata fields are present, return default medium credibility.
        # This handles empty dicts and dicts that only contain paper_id or other unrelated keys.
        has_useful_field = any(
            (k in metadata and metadata.get(k) is not None)
            for k in ("citation_count", "year", "journal")
        )
        if not has_useful_field:
            logger.debug(
                json.dumps(
                    {
                        "event": "credibility_default_medium",
                        "paper_id": str(metadata.get("paper_id", "")),
                    }
                )
            )
            return CredibilityScore(
                paper_id=str(metadata.get("paper_id", "")),
                score=50,
                tier="medium",
                rationale=default_rationale,
            )

        # Extract and coerce fields defensively
        paper_id = metadata.get("paper_id") or ""
        citation_count = cls._safe_int(metadata.get("citation_count"), default=0)
        if citation_count < 0:
            citation_count = 0

        year_raw = cls._safe_int(metadata.get("year"), default=2000)
        # Clamp future years to CURRENT_YEAR
        year = min(year_raw, cls.CURRENT_YEAR)

        journal_norm = cls._normalize_journal(metadata.get("journal"))

        # Component scoring with small integer ops
        if citation_count > 100:
            cit_score = cls.CITATION_HIGH
            cit_label = f"high citations ({citation_count})"
        elif 20 <= citation_count <= 100:
            cit_score = cls.CITATION_MEDIUM
            cit_label = f"medium citations ({citation_count})"
        else:
            cit_score = cls.CITATION_LOW
            cit_label = f"low citations ({citation_count})"

        if year >= 2022:
            rec_score = cls.RECENT_2022_PLUS
            rec_label = f"recent ({year})"
        elif 2018 <= year <= 2021:
            rec_score = cls.RECENT_2018_2021
            rec_label = f"moderately recent ({year})"
        else:
            rec_score = cls.RECENT_OLDER
            rec_label = f"older ({year})"

        venue_score = 0
        venue_label = ""
        if journal_norm:
            for token in cls.TOP_JOURNAL_SET:
                if token in journal_norm:
                    venue_score = cls.TOP_JOURNAL_SCORE
                    venue_label = "top-tier journal"
                    break
            else:
                for token in cls.PREPRINT_SET:
                    if token in journal_norm:
                        venue_score = cls.PREPRINT_PENALTY
                        venue_label = "preprint"
                        break

        contributions = {
            "citations": cit_score,
            "recency": rec_score,
            "venue": venue_score,
        }
        raw_score = (
            contributions["citations"]
            + contributions["recency"]
            + contributions["venue"]
        )
        score = cls._clamp_score(raw_score)

        if score >= cls.TIER_HIGH:
            tier = "high"
        elif score >= cls.TIER_MEDIUM:
            tier = "medium"
        else:
            tier = "low"

        parts = [cit_label, rec_label]
        if venue_label:
            parts.append(venue_label)
        rationale = ", ".join(parts)
        if len(rationale) > 100:
            rationale = rationale[:100]

        log_entry = {
            "event": "credibility_score",
            "paper_id": str(paper_id),
            "components": contributions,
            "raw_score": raw_score,
            "score": score,
            "tier": tier,
            "rationale": rationale,
        }
        logger.debug(json.dumps(log_entry))

        return CredibilityScore(
            paper_id=str(paper_id), score=score, tier=tier, rationale=rationale
        )

    @classmethod
    def score_papers(cls, papers: List[Dict[str, Any]]) -> List[CredibilityScore]:
        """Efficient bulk scoring using list comprehension.

        This remains allocation-light and keeps per-item overhead small.
        """
        return [cls.score_paper(p) for p in papers]


class ContradictionDetector:
    """
    Intelligent contradiction detection system for scientific passages.

    Analyzes passages for conflicting claims related to user queries using LLM-based analysis
    with structured output parsing and robust provider fallback.

    Features:
    - Async LLM calls with Gemini primary and Groq fallback
    - Structured output parsing with Pydantic validation
    - Tenacity-based exponential backoff retry mechanism
    - Query relevance filtering
    - Severity classification (major/minor)
    - Comprehensive logging and error handling
    """

    def __init__(
        self, retry_config: Dict[str, Any] = None, health_monitor: HealthMonitor = None
    ):
        self.parser = PydanticOutputParser(pydantic_object=List[DetectedContradiction])
        # Allow configurable retry settings for tests and operations
        rc = retry_config or {}
        self.retry_config = LLMRetryConfig(
            max_attempts=rc.get("max_attempts", 3),
            min_backoff=rc.get("min_backoff", 1.0),
            max_backoff=rc.get("max_backoff", 4.0),
        )
        # Health monitor for metrics
        self.health_monitor = health_monitor or HealthMonitor()

        # Wire tenacity-decorated call dynamically to allow configurable retry params
        # Keep base implementation in _base_call_single_llm and decorate it
        self._call_single_llm = tenacity_retry_decorator(self.retry_config)(
            self._base_call_single_llm
        )

    def _is_retryable_exception(self, exception: Exception) -> bool:
        # Delegate to shared helper
        try:
            return is_retryable_exception(exception)
        except Exception:
            return True

    async def _base_call_single_llm(
        self,
        llm,
        messages: List[Dict[str, str]],
        provider_name: str,
        request_id: str = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """Base implementation for single LLM call. This is decorated dynamically to apply retries."""
        start_time = time.time()

        try:
            # Convert messages to LangChain format
            from langchain_core.messages import HumanMessage

            langchain_messages = [
                HumanMessage(content=msg["content"]) for msg in messages
            ]

            response = await llm.ainvoke(langchain_messages)

            execution_time = time.time() - start_time

            # Record success metrics
            try:
                self.health_monitor.record_call(provider_name, True, execution_time)
                # reset provider circuit on success
                try:
                    self.health_monitor.record_provider_success(provider_name)
                except Exception:
                    pass
                tokens_used = getattr(response, "usage_metadata", {}).get(
                    "total_tokens", 0
                )
                try:
                    if tokens_used:
                        self.health_monitor.record_tokens(provider_name, tokens_used)
                except Exception:
                    pass
            except Exception:
                pass

            logger.info(
                json.dumps(
                    {
                        "event": "llm_call_success",
                        "provider": provider_name,
                        "request_id": request_id,
                        "execution_time": round(execution_time, 3),
                        "tokens_used": getattr(response, "usage_metadata", {}).get(
                            "total_tokens", 0
                        ),
                    }
                )
            )

            return response.content

        except Exception as e:
            execution_time = time.time() - start_time

            # Record failure metrics and possibly open circuit
            try:
                self.health_monitor.record_call(provider_name, False, execution_time)
                try:
                    # record provider failure which may open circuit with backoff
                    self.health_monitor.record_provider_failure(provider_name)
                except Exception:
                    pass
            except Exception:
                pass

            # Check if this exception should be retried
            if not self._is_retryable_exception(e):
                logger.error(
                    json.dumps(
                        {
                            "event": "llm_call_non_retryable",
                            "provider": provider_name,
                            "request_id": request_id,
                            "error": str(e),
                            "execution_time": round(execution_time, 3),
                        }
                    )
                )
                # Backwards-compatible short message expected by some tests and dashboards
                try:
                    logger.error(
                        "Non-retryable LLM error",
                        provider=provider_name,
                        error=str(e),
                        execution_time=round(execution_time, 3),
                    )
                except Exception:
                    # If structured params cause issues with sinks, still continue
                    logger.error("Non-retryable LLM error: %s" % str(e))
                raise

            logger.warning(
                json.dumps(
                    {
                        "event": "llm_call_retryable_error",
                        "provider": provider_name,
                        "request_id": request_id,
                        "error": str(e),
                        "execution_time": round(execution_time, 3),
                    }
                )
            )
            raise

    async def _call_llm_with_fallback(
        self,
        messages: List[Dict[str, str]],
        llm_config: Dict[str, Any],
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """
        Call LLM with primary/fallback mechanism using tenacity retry.

        Args:
            messages: List of message dicts with 'role' and 'content'
            llm_config: Config dict with API keys
            temperature: LLM temperature
            max_tokens: Max tokens for response

        Returns:
            LLM response content

        Raises:
            Exception: If both providers fail after retries
        """
        # Instantiate real LLM clients only when API keys are provided to avoid
        # creating external clients during unit tests where we monkeypatch
        # _call_single_llm.
        primary_llm = None
        fallback_llm = None
        try:
            if llm_config.get("GOOGLE_API_KEY") or llm_config.get("FORCE_REAL_LLMS"):
                primary_llm = ChatGoogleGenerativeAI(
                    model=os.getenv("LLM_MODEL_PRIMARY"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    google_api_key=llm_config.get("GOOGLE_API_KEY"),
                )

            if llm_config.get("GROQ_API_KEY") or llm_config.get("FORCE_REAL_LLMS"):
                fallback_llm = ChatGroq(
                    model=llm_config.get("LLM_MODEL_FALLBACK", "llama3-8b-8192"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=llm_config.get("GROQ_API_KEY"),
                )
        except Exception as e:
            # If client construction fails (e.g., missing creds in test env),
            # log and continue with None placeholders so monkeypatched callables
            # can handle the LLM call.
            logger.warning(
                json.dumps({"event": "llm_client_init_failed", "error": str(e)})
            )

        providers = [("gemini", primary_llm), ("groq", fallback_llm)]
        request_id = (
            llm_config.get("request_id") if isinstance(llm_config, dict) else None
        )

        for provider_name, llm in providers:
            # Skip providers currently marked as down by circuit breaker
            try:
                if not self.health_monitor.is_provider_available(provider_name):
                    logger.warning(
                        json.dumps(
                            {
                                "event": "provider_skipped_circuit_open",
                                "provider": provider_name,
                                "request_id": request_id,
                            }
                        )
                    )
                    # count as fallback skip
                    try:
                        self.health_monitor.record_fallback()
                    except Exception:
                        pass
                    continue
            except Exception:
                # if health monitor fails, proceed to attempt call
                pass
            try:
                logger.info(
                    json.dumps(
                        {
                            "event": "attempting_llm_call",
                            "provider": provider_name,
                            "request_id": request_id,
                        }
                    )
                )

                response = await self._call_single_llm(
                    llm=llm,
                    messages=messages,
                    provider_name=provider_name,
                    request_id=request_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return response

            except Exception as e:
                logger.warning(
                    json.dumps(
                        {
                            "event": "provider_failed_trying_fallback",
                            "failed_provider": provider_name,
                            "error": str(e),
                            "request_id": request_id,
                        }
                    )
                )

                # Record fallback event
                try:
                    self.health_monitor.record_fallback()
                except Exception:
                    pass

                if provider_name == providers[-1][0]:  # Last provider
                    logger.error(
                        json.dumps(
                            {
                                "event": "all_providers_failed",
                                "final_error": str(e),
                                "request_id": request_id,
                            }
                        )
                    )
                    raise

                continue  # Try next provider

    async def detect_contradictions(
        self,
        query: str,
        relevant_passages: List[Dict[str, Any]],
        llm_config: Dict[str, Any],
    ) -> List[DetectedContradiction]:
        """
        Detect contradictions in scientific passages related to the user query.

        Args:
            query: User's scientific question
            relevant_passages: List of passage dicts with paper_id, text, paper_title
            llm_config: LLM configuration with API keys

        Returns:
            List of detected contradictions, empty list if none found or errors
        """
        start_time = time.time()
        query_truncated = len(query) > 100

        # Truncate query if too long for logging
        log_query = query[:100] + "..." if query_truncated else query

        logger.info(
            "Starting contradiction detection",
            query=log_query,
            query_truncated=query_truncated,
            passage_count=len(relevant_passages),
        )

        try:
            # Format passages for LLM with token optimization
            passages_text = self._format_passages(relevant_passages)

            # Create the prompt
            prompt = CONTRADICTION_DETECTION_PROMPT.format(
                query=query, passages=passages_text
            )

            messages = [{"role": "user", "content": prompt}]

            # Call LLM
            response_content = await self._call_llm_with_fallback(
                messages=messages,
                llm_config=llm_config,
                temperature=0.2,
                max_tokens=2000,
            )

            # Parse response
            contradictions = self._parse_response(response_content)

            execution_time = time.time() - start_time

            logger.info(
                "Contradiction detection completed",
                contradictions_found=len(contradictions),
                passages_analyzed=len(relevant_passages),
                execution_time=round(execution_time, 3),
                success=True,
            )

            return contradictions

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Contradiction detection failed",
                error=str(e),
                passages_analyzed=len(relevant_passages),
                execution_time=round(execution_time, 3),
                success=False,
            )
            return []

    def _format_passages(self, passages: List[Dict[str, Any]]) -> str:
        """Format passages for LLM analysis with token limit optimization."""
        # Estimate tokens: roughly 4 chars per token
        MAX_TOTAL_TOKENS = 12000  # Leave room for prompt and response
        MAX_PASSAGE_TOKENS = 800  # Max tokens per passage
        RESERVED_TOKENS = 2000  # For prompt template and metadata

        formatted = []
        total_tokens = 0

        for i, passage in enumerate(passages[:10], 1):  # Max 10 passages
            paper_id = passage.get("paper_id", "unknown")
            text = passage.get("text", "")
            title = passage.get("paper_title", "Unknown Title")

            # Optimize text extraction to preserve key claims
            optimized_text = self._optimize_passage_text(text, MAX_PASSAGE_TOKENS)

            # Estimate tokens for this passage
            passage_text = (
                f"Passage {i} (Paper ID: {paper_id}, Title: {title}):\n{optimized_text}"
            )
            passage_tokens = len(passage_text) // 4  # Rough token estimate

            # Check if adding this passage would exceed total limit
            if total_tokens + passage_tokens > MAX_TOTAL_TOKENS - RESERVED_TOKENS:
                logger.warning(
                    "Stopping passage formatting due to token limit",
                    passages_processed=i - 1,
                    total_tokens_estimate=total_tokens,
                    max_tokens=MAX_TOTAL_TOKENS,
                )
                break

            formatted.append(passage_text)
            total_tokens += passage_tokens

        logger.info(
            "Passages formatted for LLM",
            passages_formatted=len(formatted),
            total_tokens_estimate=total_tokens,
        )

        return "\n\n".join(formatted)

    def _optimize_passage_text(self, text: str, max_tokens: int) -> str:
        """
        Optimize passage text to fit within token limits while preserving key claims.

        Strategy:
        1. Extract sentences containing scientific claims (methods, results, conclusions)
        2. Prioritize sentences with quantitative data, comparisons, and causal language
        3. Truncate if still too long, preferring start and end of text
        """
        if not text:
            return ""

        # Estimate current tokens
        current_tokens = len(text) // 4
        if current_tokens <= max_tokens:
            return text

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # Score sentences for importance
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence_importance(sentence)
            scored_sentences.append((sentence, score))

        # Sort by importance (highest first)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Build optimized text
        optimized_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Convert tokens to chars

        for sentence, score in scored_sentences:
            if total_chars + len(sentence) > max_chars:
                break
            optimized_parts.append(sentence)
            total_chars += len(sentence)

        # Sort back to original order for coherence
        optimized_text = " ".join(optimized_parts)

        # If we have very little content, fall back to truncated original
        if len(optimized_text) < 100:
            optimized_text = text[:max_chars]

        # Ensure we don't exceed the limit
        if len(optimized_text) > max_chars:
            optimized_text = optimized_text[:max_chars]

        return optimized_text.strip()

    def _score_sentence_importance(self, sentence: str) -> float:
        """Score a sentence for scientific importance."""
        score = 0.0
        text_lower = sentence.lower()

        # Scientific claim indicators
        claim_indicators = [
            "result",
            "finding",
            "conclusion",
            "show",
            "demonstrate",
            "indicate",
            "suggest",
            "evidence",
            "data",
            "study",
            "research",
            "analysis",
            "significant",
            "effect",
            "impact",
            "influence",
            "association",
            "correlation",
            "relationship",
            "difference",
            "comparison",
        ]

        for indicator in claim_indicators:
            if indicator in text_lower:
                score += 1.0

        # Quantitative data (numbers, percentages)
        if re.search(r"\d+(\.\d+)?%?", sentence):
            score += 2.0

        # Comparative language
        comparative_words = [
            "more",
            "less",
            "higher",
            "lower",
            "greater",
            "smaller",
            "increase",
            "decrease",
        ]
        for word in comparative_words:
            if word in text_lower:
                score += 1.5

        # Causal language
        causal_words = [
            "cause",
            "lead",
            "result",
            "due",
            "because",
            "therefore",
            "thus",
            "consequently",
        ]
        for word in causal_words:
            if word in text_lower:
                score += 1.5

        # Length penalty (prefer concise sentences)
        word_count = len(sentence.split())
        if word_count > 50:
            score -= 0.5
        elif word_count < 10:
            score -= 0.2

        return max(score, 0.1)  # Minimum score

    def _parse_response(self, response_content: str) -> List[DetectedContradiction]:
        """Parse LLM response into DetectedContradiction objects."""
        try:
            # Try to parse as JSON first
            import json

            parsed_data = json.loads(response_content.strip())

            # Validate it's a list
            if not isinstance(parsed_data, list):
                raise ValueError("Response is not a list")

            # Parse each item
            contradictions = []
            for item in parsed_data:
                try:
                    contradiction = DetectedContradiction(**item)
                    contradictions.append(contradiction)
                except ValidationError as ve:
                    logger.warning(
                        json.dumps(
                            {
                                "event": "contradiction_validation_error",
                                "error": str(ve),
                                "item": item,
                            }
                        )
                    )
                    continue

            return contradictions[:3]  # Max 3 contradictions

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                json.dumps(
                    {
                        "event": "response_parse_error",
                        "error": str(e),
                        "response_preview": response_content[:200],
                    }
                )
            )
            # Return empty list on parse failure
            return []
