"""RAG service for stages 7-10 of the pipeline.

Responsibilities:
7. Hybrid Retrieval
8. Cross-Encoder Reranking
9. Synthesis
10. Verification
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from config.config import Settings
from models.state import Passage
from services.exceptions import (
    RerankingError,
    RetrievalError,
    SynthesisError,
    VerificationError,
)

if TYPE_CHECKING:
    from agents.synthesis_agent import SynthesisAgent
    from retrieval.llamaindex_retriever import LlamaIndexRetriever
    from retrieval.reranker import CrossEncoderReranker
    from utils.citation_verifier import CitationVerifier


@dataclass
class RAGResult:
    """Result of the RAG phase."""

    passages: List[Passage]
    answer: Optional[str]
    citation_verification: Optional[Dict[str, Any]]
    timing_breakdown: Dict[str, float]


class RAGService:
    """Service for retrieval, reranking, synthesis, and verification."""

    def __init__(self, embedding_model: Any, settings: Settings):
        """Initialize the RAG service.

        Args:
            embedding_model: Pre-loaded embedding model for retrieval.
            settings: Application settings.
        """
        self.embedding_model = embedding_model
        self.settings = settings

        # Initialize components immediately
        # Note: We keep imports inside __init__ to avoid circular deps and maintain safe DLL import order
        from agents.synthesis_agent import SynthesisAgent
        from retrieval.llamaindex_retriever import LlamaIndexRetriever
        from retrieval.reranker import CrossEncoderReranker
        from utils.citation_verifier import CitationVerifier

        self.retriever = LlamaIndexRetriever(
            embedding_model=embedding_model,
            top_k=settings.RAG_RETRIEVAL_TOP_K,
        )

        try:
            self.reranker = CrossEncoderReranker()
        except (ImportError, OSError) as e:
            logger.warning(
                f"Could not import CrossEncoderReranker: {e}. Reranking will fail if attempted."
            )
            self.reranker = None

        self.synthesis_agent = SynthesisAgent(settings=settings)
        self.verifier = CitationVerifier()

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
            logger.info(
                "Diversity filter too aggressive, filling {} spots with skipped passages",
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

        if self.embedding_model is None:
            logger.warning("⚠️ Embedding model not available - skipping RAG stages.")
            return RAGResult(
                passages=[],
                answer=None,
                citation_verification=None,
                timing_breakdown=timing_breakdown,
            )

        # Stage 7: Hybrid Retrieval
        logger.info(
            "🔍 Stage 7: Hybrid Retrieval - indexing {} chunks",
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

        # Stage 8: Cross-Encoder Reranking
        logger.info(
            "⚖️ Stage 8: Cross-Encoder Reranking - reranking {} passages",
            len(passages),
        )
        stage_start = time.perf_counter()
        try:
            if self.reranker:
                # Rerank a larger pool (all retrieved) to allow for diversity filtering
                reranked_passages = await asyncio.to_thread(
                    self.reranker.rerank, query, passages, top_k=len(passages)
                )

                # Apply diversity filter
                logger.info(
                    "Applying diversity filter: selecting top {} with max 3 per paper",
                    self.settings.RAG_RERANK_TOP_K,
                )
                passages = self._select_diverse_top_k(
                    reranked_passages, k=self.settings.RAG_RERANK_TOP_K
                )

                timing_breakdown["reranking"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ Reranking completed in {:.3f}s - selected top {} passages",
                    timing_breakdown["reranking"],
                    len(passages),
                )
            else:
                logger.warning(
                    f"⚠️ Reranker not available - skipping reranking and taking top {self.settings.RAG_RERANK_TOP_K}"
                )
                passages = passages[: self.settings.RAG_RERANK_TOP_K]
                timing_breakdown["reranking"] = 0.0
        except Exception as e:
            logger.exception("Reranking failed")
            raise RerankingError(f"Failed to rerank passages: {str(e)}") from e

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
                    "✅ Verification completed in {:.3f}s - score: {}",
                    timing_breakdown["verification"],
                    verification_result.get("score", 0),
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
