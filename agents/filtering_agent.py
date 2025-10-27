from rank_bm25 import BM25Okapi
from sentence_transformers import util
import numpy as np
import logging
import time
from typing import List, Tuple
from fastapi import Request
from models.state import State, Paper

logger = logging.getLogger(__name__)


class FilteringAgent:
    """
    Two-pass paper filtering: BM25 (keyword) + Semantic Embeddings (meaning).

    Architecture:
    1. Pass 1 - BM25 Filtering (100-300 → 50 papers, ~40ms):
       - Tokenize title + abstract for each paper
       - Score against query using BM25 algorithm
       - Select top 50 papers by BM25 score

    2. Pass 2 - Semantic Reranking (50 → 25 papers, ~150ms):
       - Embed abstracts using cached SentenceTransformer model
       - Calculate cosine similarity with query embedding
       - Combine semantic score with metadata (citations, recency, type)
       - Select top 25 papers by composite score

    Performance: ~200ms total for 300 papers (10x faster than embedding all 300)
    """

    def __init__(self, request: Request):
        """
        Initialize FilteringAgent with cached embedding model.

        Args:
            request: FastAPI Request object with app.state.embedding_model
        """
        # Access cached embedding model from app.state
        if (
            hasattr(request.app.state, "embedding_model")
            and request.app.state.embedding_model
        ):
            self.embedding_model = request.app.state.embedding_model
        else:
            raise RuntimeError(
                "Embedding model not found in app.state. "
                "Ensure FastAPI startup event loads model."
            )

    def filter_candidates(self, state: State) -> State:
        """
        Filter papers using two-pass BM25 + semantic embeddings strategy.

        Args:
            state: LangGraph State with papers_metadata (deduplicated papers)

        Returns:
            Updated State with filtered_papers (top 25) and filtering_scores (metrics)
        """
        papers = state.papers_metadata
        query = state.optimized_query or state.original_query

        paper_count = len(papers)
        logger.info(f"FilteringAgent: Filtering {paper_count} papers → top 25")

        if paper_count == 0:
            logger.warning("FilteringAgent: No papers to filter")
            state.filtered_papers = []
            state.filtering_scores = {
                "error": "No papers to filter",
                "initial_count": 0,
                "bm25_filtered_count": None,
                "final_count": 0,
                "bm25_time_ms": 0.0,
                "semantic_time_ms": 0.0,
                "total_time_ms": 0.0,
                "avg_bm25_score": 0.0,
                "avg_composite_score": 0.0,
            }
            return state

        # If we have ≤25 papers, skip filtering
        if paper_count <= 25:
            logger.info(
                f"FilteringAgent: Only {paper_count} papers, skipping filtering"
            )
            state.filtered_papers = papers
            state.filtering_scores = {
                "skipped": True,
                "reason": "Insufficient papers for filtering",
                "initial_count": paper_count,
                "bm25_filtered_count": None,
                "final_count": paper_count,
                "bm25_time_ms": 0.0,
                "semantic_time_ms": 0.0,
                "total_time_ms": 0.0,
                "avg_bm25_score": 0.0,
                "avg_composite_score": 0.0,
            }
            return state

        # Pass 1: BM25 Filtering (papers → top 50)
        start_bm25 = time.time()
        top_50_papers, bm25_scores = self._bm25_filter(papers, query, target_count=50)
        bm25_time = time.time() - start_bm25

        # Pass 2: Semantic Reranking (top 50 → top 25)
        start_semantic = time.time()
        top_25_papers, composite_scores = self._semantic_rerank(
            top_50_papers, query, target_count=25
        )
        semantic_time = time.time() - start_semantic

        # Update state
        state.filtered_papers = top_25_papers
        state.filtering_scores = {
            "initial_count": paper_count,
            "bm25_filtered_count": len(top_50_papers),
            "final_count": len(top_25_papers),
            "bm25_time_ms": round(bm25_time * 1000, 1),
            "semantic_time_ms": round(semantic_time * 1000, 1),
            "total_time_ms": round((bm25_time + semantic_time) * 1000, 1),
            "avg_bm25_score": (
                float(np.mean(bm25_scores[:50])) if len(bm25_scores) > 0 else 0.0
            ),
            "avg_composite_score": (
                float(np.mean(composite_scores)) if len(composite_scores) > 0 else 0.0
            ),
        }

        logger.info(
            f"FilteringAgent: {paper_count} → {len(top_25_papers)} papers "
            f"(BM25: {bm25_time*1000:.1f}ms, Semantic: {semantic_time*1000:.1f}ms)"
        )

        return state

    def _bm25_filter(
        self, papers: List[Paper], query: str, target_count: int = 50
    ) -> Tuple[List[Paper], np.ndarray]:
        """
        Pass 1: BM25 keyword-based filtering.

        Uses BM25 algorithm to score papers based on term frequency overlap
        between query and paper title+abstract.

        Args:
            papers: List of papers to filter
            query: Search query
            target_count: Number of papers to keep (default: 50)

        Returns:
            Tuple of (top_papers, bm25_scores_array)
        """
        # Tokenize corpus (title + abstract for each paper)
        corpus = []
        for paper in papers:
            # Combine title and abstract (abstract may be None)
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract

            # Tokenize (lowercase, split on whitespace)
            tokens = text.lower().split()
            corpus.append(tokens)

        # Build BM25 index
        bm25 = BM25Okapi(corpus)

        # Score all papers against query
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Get top N papers by BM25 score
        top_indices = np.argsort(bm25_scores)[-target_count:][::-1]  # Descending order
        top_papers = [papers[i] for i in top_indices]

        return top_papers, bm25_scores

    def _semantic_rerank(
        self, papers: List[Paper], query: str, target_count: int = 25
    ) -> Tuple[List[Paper], List[float]]:
        """
        Pass 2: Semantic embedding-based reranking with metadata scoring.

        Combines:
        - Semantic similarity (query vs. abstract embeddings)
        - Citation count (credibility signal)
        - Recency (prioritize recent publications)
        - Publication type (boost reviews for context)

        Args:
            papers: List of papers to rerank (typically top 50 from BM25)
            query: Search query
            target_count: Number of papers to keep (default: 25)

        Returns:
            Tuple of (top_papers, composite_scores_list)
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        # Embed abstracts (or titles if abstract missing)
        abstracts = []
        for paper in papers:
            text = paper.abstract if paper.abstract else paper.title
            abstracts.append(text)

        abstract_embeddings = self.embedding_model.encode(
            abstracts,
            convert_to_tensor=True,
            batch_size=32,  # Batch for efficiency
            show_progress_bar=False,
        )

        # Calculate semantic similarity (cosine similarity)
        semantic_scores = (
            util.cos_sim(query_embedding, abstract_embeddings)[0].cpu().numpy()
        )

        # Calculate composite scores
        composite_scores = []
        for i, paper in enumerate(papers):
            # Semantic similarity (0-1)
            semantic_score = float(semantic_scores[i])

            # Citation credibility (normalize to 0-1, cap at 500 citations)
            citation_score = min(paper.citation_count / 500.0, 1.0)

            # Recency score (2025 = 1.0, decays 0.1 per year)
            current_year = 2025
            recency_score = max(1.0 - (current_year - (paper.year or 2020)) * 0.1, 0.0)

            # Publication type boost (reviews get 20% boost)
            type_boost = 1.2 if paper.type == "review" else 1.0

            # Weighted composite score
            composite = (
                0.5 * semantic_score  # Semantic similarity is most important
                + 0.25 * citation_score  # Citations indicate credibility
                + 0.25 * recency_score  # Recency for up-to-date info
            ) * type_boost

            composite_scores.append(composite)

        # Sort by composite score and select top N
        sorted_indices = np.argsort(composite_scores)[-target_count:][::-1]
        top_papers = [papers[i] for i in sorted_indices]
        top_scores = [composite_scores[i] for i in sorted_indices]

        return top_papers, top_scores
