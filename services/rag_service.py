"""RAG service for stages 7-9 of the pipeline.

Responsibilities:
7. BM25 Retrieval
8. Synthesis
9. Verification
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from config.config import Settings
from models.state import Passage
from services.exceptions import RetrievalError, SynthesisError, VerificationError

if TYPE_CHECKING:
    from agents.synthesis_agent import SynthesisAgent
    from retrieval.bm25_retriever import BM25Retriever
    from utils.citation_verifier import CitationVerifier


@dataclass
class RAGResult:
    """Result of the RAG phase."""

    passages: List[Passage]
    answer: Optional[str]
    citation_verification: Optional[Dict[str, Any]]
    timing_breakdown: Dict[str, float]


class RAGService:
    """Service for retrieval, synthesis, and verification."""

    def __init__(self, settings: Settings):
        """Initialize the RAG service.

        Args:
            settings: Application settings.
        """
        self.settings = settings

        # Initialize components immediately
        # Note: We keep imports inside __init__ to avoid circular deps
        from agents.synthesis_agent import SynthesisAgent
        from retrieval.bm25_retriever import BM25Retriever
        from utils.citation_verifier import CitationVerifier

        self.retriever = BM25Retriever(top_k=settings.RAG_RETRIEVAL_TOP_K)
        self.synthesis_agent = SynthesisAgent(settings=settings)
        self.verifier = CitationVerifier()

    def _select_diverse_top_k(
        self, passages: List[Any], k: int, max_per_paper: int = 3
    ) -> List[Any]:
        """Select top k passages with diversity constraint (Grouped Top-K).

        Args:
            passages: List of passages sorted by score (descending).
            k: Target number of passages to return.
            max_per_paper: Maximum number of passages allowed from a single paper.

        Returns:
            List of selected passages.
        """
        if not passages:
            return []

        selected_passages = []
        paper_counts = defaultdict(int)
        skipped_passages = []

        # First pass: Select satisfying the max_per_paper constraint
        for passage in passages:
            if len(selected_passages) >= k:
                break

            # Extract paper_id safely
            paper_id = "unknown"
            # Handle NodeWithScore object from llama_index
            if hasattr(passage, "node") and hasattr(passage.node, "metadata"):
                paper_id = passage.node.metadata.get("paper_id", "unknown")

            if paper_counts[paper_id] < max_per_paper:
                selected_passages.append(passage)
                paper_counts[paper_id] += 1
            else:
                skipped_passages.append(passage)

        # Fallback: If we don't have enough passages, fill with skipped ones
        if len(selected_passages) < k and skipped_passages:
            needed = k - len(selected_passages)
            if needed > k // 2:
                logger.warning(
                    "Diversity constraint left {}/{} spots unfilled — filling from overflow",
                    needed,
                    k,
                )
            else:
                logger.debug(
                    "Diversity constraint filled {} overflow spots",
                    needed,
                )
            selected_passages.extend(skipped_passages[:needed])

        return selected_passages

    async def answer(self, query: str, chunks: List[Any]) -> RAGResult:
        """Execute the RAG phase (Stages 7-10).

        Args:
            query: The optimized query string (final_rephrase).
            chunks: List of processed text chunks.

        Returns:
            RAGResult containing answer and verification details.
        """
        timing_breakdown = {}
        passages = []
        answer = None
        verification_result = None

        if not chunks:
            logger.warning("⚠️ No chunks available for retrieval")
            return RAGResult(
                passages=[],
                answer=None,
                citation_verification=None,
                timing_breakdown=timing_breakdown,
            )

        # Stage 7: BM25 Retrieval
        logger.info(
            "🔍 Stage 7: BM25 Retrieval - indexing {} chunks",
            len(chunks),
        )
        stage_start = time.perf_counter()
        try:
            # Retrieve
            retrieved_nodes = await asyncio.to_thread(
                self.retriever.retrieve, query, chunks
            )

            # Convert nodes to simple dicts/objects if needed,
            # but the original code seems to use the nodes directly or convert them.
            # Let's assume retrieved_nodes are compatible with what reranker expects.
            passages = retrieved_nodes

            timing_breakdown["retrieval"] = time.perf_counter() - stage_start
            logger.info(
                "✅ Retrieval completed in {:.3f}s - retrieved {} candidates",
                timing_breakdown["retrieval"],
                len(passages),
            )
        except Exception as e:
            logger.exception("Retrieval failed")
            raise RetrievalError(f"Failed to retrieve passages: {str(e)}") from e

        if not passages:
            logger.warning("⚠️ No passages retrieved")
            return RAGResult(
                passages=[],
                answer=None,
                citation_verification=None,
                timing_breakdown=timing_breakdown,
            )

        logger.info(
            "Applying diversity filter: selecting top {} with max 3 per paper",
            self.settings.RAG_PASSAGES_TOP_K,
        )
        passages = self._select_diverse_top_k(
            passages, k=self.settings.RAG_PASSAGES_TOP_K
        )
        logger.info("✅ Selected {} passages after diversity filter", len(passages))

        # Convert to Pydantic models for result
        final_passages = []
        for p in passages:
            final_passages.append(
                Passage(
                    paper_id=p.node.metadata.get("paper_id", "unknown"),
                    text=p.node.get_content(),
                    score=p.score,
                    metadata=p.node.metadata,
                )
            )

        # Stage 9: Synthesis
        logger.info("🤖 Stage 9: Synthesis - generating answer")
        stage_start = time.perf_counter()
        try:
            answer = await self.synthesis_agent.synthesize(query, passages)
            timing_breakdown["synthesis"] = time.perf_counter() - stage_start
            logger.info(
                "✅ Synthesis completed in {:.3f}s", timing_breakdown["synthesis"]
            )
        except Exception as e:
            timing_breakdown["synthesis"] = time.perf_counter() - stage_start
            logger.error(f"Synthesis failed: {e}")
            # Graceful degradation: return passages without answer
            answer = None
            # We do NOT raise SynthesisError here to allow the pipeline to return passages

        # Stage 10: Verification
        if answer:
            logger.info("✅ Stage 10: Verification - checking citations")
            stage_start = time.perf_counter()
            try:
                verification_result = self.verifier.verify(answer, passages)
                timing_breakdown["verification"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ Verification completed in {:.3f}s - valid={}, citations_found={}",
                    timing_breakdown["verification"],
                    verification_result.get("valid"),
                    len(verification_result.get("citations_found", [])),
                )
            except Exception as e:
                logger.warning("⚠️ Verification failed: {}", str(e))
                # Don't fail the pipeline for verification error
                verification_result = {"error": str(e)}

        return RAGResult(
            passages=final_passages,
            answer=answer,
            citation_verification=verification_result,
            timing_breakdown=timing_breakdown,
        )
