"""ResearchService: iterative multi-round research loop.

Orchestrates PlanningAgent → DiscoveryService → IngestionService →
RAGService → ReflectionAgent across up to MAX_ROUNDS iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

from loguru import logger

from agents.planning_agent import PlanningAgent, PlanResult
from agents.reflection_agent import ReflectionAgent, ReflectionResult

MAX_ROUNDS: int = 3
PDF_TARGETS: dict[int, int] = {1: 3, 2: 2, 3: 1}  # papers to ingest per round


@dataclass
class ResearchResult:
    """Final result returned from ResearchService.research()."""

    answer: Optional[str]
    research_brief: str
    rounds_completed: int
    searched_queries: List[str]
    total_chunks: int
    timing_breakdown: dict[str, float] = field(default_factory=dict)


class ResearchService:
    """Stateful multi-round research loop.

    Accepts pre-constructed DiscoveryService, IngestionService, and RAGService
    so the caller (QueryService) controls their lifecycle.
    """

    def __init__(
        self,
        discovery_service: Any,
        ingestion_service: Any,
        rag_service: Any,
    ) -> None:
        self.discovery = discovery_service
        self.ingestion = ingestion_service
        self.rag = rag_service
        self._planning_agent = PlanningAgent()
        self._reflection_agent = ReflectionAgent()

    async def research(self, query: str) -> ResearchResult:
        """Run the multi-round research loop for a user query.

        Args:
            query: Raw user research question.

        Returns:
            ResearchResult with the best answer produced across all rounds.
        """
        overall_start = time.perf_counter()
        timing: dict[str, float] = {}

        # --- Phase: Planning (once, never changes) ---
        t0 = time.perf_counter()
        plan: PlanResult = await self._planning_agent.plan(query)
        timing["planning"] = time.perf_counter() - t0
        research_brief = plan["research_brief"]
        logger.info(
            "ResearchService: brief ({} chars), initial_query='{}'",
            len(research_brief),
            plan["initial_query"],
        )

        # --- Persistent state ---
        seen_paper_ids: Set[str] = set()
        searched_queries: List[str] = []
        current_chunks: List[Any] = []
        current_answer: Optional[str] = None
        reflection: Optional[ReflectionResult] = None

        for round_num in range(1, MAX_ROUNDS + 1):
            logger.info("ResearchService: starting round {}/{}", round_num, MAX_ROUNDS)

            # Determine which query to search this round
            if round_num == 1:
                query_to_search = plan["initial_query"]
            else:
                assert reflection is not None
                if reflection["next_query"] is None:
                    logger.info(
                        "ResearchService: round {} — no next_query from reflection, breaking",
                        round_num,
                    )
                    break
                query_to_search = reflection["next_query"]

            searched_queries.append(query_to_search)

            # --- Discovery ---
            t0 = time.perf_counter()
            try:
                discovery_result = await self.discovery.discover(query_to_search)
            except Exception as e:
                logger.exception(
                    "ResearchService: discovery failed on round {}: {}", round_num, e
                )
                break
            timing[f"discovery_round_{round_num}"] = time.perf_counter() - t0

            # Filter out already-seen papers (Phase 5 will push this into DiscoveryService)
            pdf_target = PDF_TARGETS.get(round_num, 1)
            new_finalists = [
                p for p in discovery_result.finalists if p.paperId not in seen_paper_ids
            ][:pdf_target]

            if not new_finalists:
                logger.warning(
                    "ResearchService: round {} returned 0 new papers — breaking early",
                    round_num,
                )
                break

            # Track seen papers
            for p in new_finalists:
                seen_paper_ids.add(p.paperId)

            # --- Ingestion ---
            t0 = time.perf_counter()
            try:
                ingestion_result = await self.ingestion.ingest(new_finalists, query)
            except Exception as e:
                logger.exception(
                    "ResearchService: ingestion failed on round {}: {}", round_num, e
                )
                break
            timing[f"ingestion_round_{round_num}"] = time.perf_counter() - t0

            new_chunks = ingestion_result.chunks
            if not new_chunks:
                logger.warning(
                    "ResearchService: round {} produced 0 chunks — breaking early",
                    round_num,
                )
                break

            # Merge corpus — grows each round
            current_chunks = current_chunks + new_chunks
            logger.info(
                "ResearchService: round {} merged corpus = {} chunks total",
                round_num,
                len(current_chunks),
            )

            # --- Synthesis (always use final_rephrase, never the gap query) ---
            t0 = time.perf_counter()
            try:
                rag_result = await self.rag.answer(
                    plan["final_rephrase"], current_chunks
                )
                if rag_result.answer is not None:
                    current_answer = rag_result.answer
                else:
                    logger.warning(
                        "ResearchService: synthesis returned None on round {} — keeping previous answer",
                        round_num,
                    )
            except Exception as e:
                logger.exception(
                    "ResearchService: synthesis failed on round {}: {}", round_num, e
                )
                # Do NOT overwrite current_answer with None
            timing[f"synthesis_round_{round_num}"] = time.perf_counter() - t0

            # --- Reflection ---
            if current_answer is None:
                logger.warning(
                    "ResearchService: no answer yet after round {}, skipping reflection",
                    round_num,
                )
                break

            t0 = time.perf_counter()
            reflection = await self._reflection_agent.evaluate(
                query=query,
                research_brief=research_brief,
                current_answer=current_answer,
                searched_queries=searched_queries,
            )
            timing[f"reflection_round_{round_num}"] = time.perf_counter() - t0

            logger.info(
                "ResearchService: round {} complete — is_complete={}, gap='{}'",
                round_num,
                reflection["is_complete"],
                reflection["gap_description"][:80],
            )

            if reflection["is_complete"]:
                logger.info(
                    "ResearchService: research complete after round {}", round_num
                )
                break

        timing["total"] = time.perf_counter() - overall_start
        logger.info(
            "ResearchService: finished {} round(s) in {:.2f}s — {} chunks, answer={}",
            len(searched_queries),
            timing["total"],
            len(current_chunks),
            "yes" if current_answer else "no",
        )

        return ResearchResult(
            answer=current_answer,
            research_brief=research_brief,
            rounds_completed=len(searched_queries),
            searched_queries=searched_queries,
            total_chunks=len(current_chunks),
            timing_breakdown=timing,
        )


__all__ = ["ResearchService", "ResearchResult", "MAX_ROUNDS"]
