"""RAG service for stages 7-10 of the pipeline.

Responsibilities:
7. Hybrid Retrieval
8. Cross-Encoder Reranking
9. Synthesis
10. Verification
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from services.exceptions import (
    RetrievalError,
    RerankingError,
    SynthesisError,
    VerificationError,
)
from config.config import RAG_RETRIEVAL_TOP_K, RAG_RERANK_TOP_K

if TYPE_CHECKING:
    from retrieval.llamaindex_retriever import LlamaIndexRetriever
    from retrieval.reranker import CrossEncoderReranker
    from agents.synthesis_agent import SynthesisAgent
    from utils.citation_verifier import CitationVerifier


@dataclass
class RAGResult:
    """Result of the RAG phase."""

    passages: List[Any]
    answer: Optional[str]
    citation_verification: Optional[Dict[str, Any]]
    timing_breakdown: Dict[str, float]


class RAGService:
    """Service for retrieval, reranking, synthesis, and verification."""

    def __init__(self, embedding_model: Any = None):
        """Initialize the RAG service.

        Args:
            embedding_model: Pre-loaded embedding model for retrieval.
        """
        self.embedding_model = embedding_model
        self._retriever = None
        self._synthesis_agent = None
        self._verifier = None

        # Pre-load CrossEncoderReranker during initialization to avoid ~12s latency
        # on first request. Import here to maintain safe DLL import order on Windows.
        try:
            from retrieval.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        except (ImportError, OSError) as e:
            logger.warning(
                f"Could not import CrossEncoderReranker: {e}. Reranking will fail if attempted."
            )
            self._reranker = None

    def _get_retriever(self) -> Optional["LlamaIndexRetriever"]:
        """Lazy initialization of hybrid retriever.

        Returns None if embedding_model is not available.
        """
        if self._retriever is None and self.embedding_model is not None:
            from retrieval.llamaindex_retriever import LlamaIndexRetriever

            self._retriever = LlamaIndexRetriever(
                embedding_model=self.embedding_model,
                top_k=RAG_RETRIEVAL_TOP_K,  # Before reranking
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
            # Handle dict or other objects
            elif hasattr(passage, "metadata"):
                paper_id = passage.metadata.get("paper_id", "unknown")
            elif isinstance(passage, dict):
                paper_id = passage.get("metadata", {}).get("paper_id", "unknown")

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
            retriever = self._get_retriever()
            if retriever:
                # Retrieve
                retrieved_nodes = await asyncio.to_thread(
                    retriever.retrieve, query, chunks
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
            else:
                raise RetrievalError("Retriever could not be initialized")
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
            reranker = self._get_reranker()
            if reranker:
                # Rerank a larger pool (all retrieved) to allow for diversity filtering
                reranked_passages = await asyncio.to_thread(
                    reranker.rerank, query, passages, top_k=len(passages)
                )

                # Apply diversity filter
                logger.info(
                    "Applying diversity filter: selecting top {} with max 3 per paper",
                    RAG_RERANK_TOP_K,
                )
                passages = self._select_diverse_top_k(
                    reranked_passages, k=RAG_RERANK_TOP_K
                )

                timing_breakdown["reranking"] = time.perf_counter() - stage_start
                logger.info(
                    "✅ Reranking completed in {:.3f}s - selected top {} passages",
                    timing_breakdown["reranking"],
                    len(passages),
                )
            else:
                logger.warning(
                    f"⚠️ Reranker not available - skipping reranking and taking top {RAG_RERANK_TOP_K}"
                )
                passages = passages[:RAG_RERANK_TOP_K]
                timing_breakdown["reranking"] = 0.0
        except Exception as e:
            logger.exception("Reranking failed")
            raise RerankingError(f"Failed to rerank passages: {str(e)}") from e

        # Stage 9: Synthesis
        logger.info("🤖 Stage 9: Synthesis - generating answer")
        stage_start = time.perf_counter()
        try:
            synthesis_agent = self._get_synthesis_agent()
            answer = await synthesis_agent.synthesize(query, passages)
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
                verifier = self._get_verifier()
                verification_result = verifier.verify(answer, passages)
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
            passages=passages,
            answer=answer,
            citation_verification=verification_result,
            timing_breakdown=timing_breakdown,
        )
