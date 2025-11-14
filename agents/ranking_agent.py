"""Ranking agent for scoring and ranking Semantic Scholar papers.

Provides multi-factor paper ranking combining semantic similarity, authority,
impact, and recency factors. Returns top 5-10 finalist papers for downstream
PDF processing.

Usage example:
    from agents.ranking_agent import RankingAgent
    import numpy as np

    agent = RankingAgent()
    papers = [...]  # List of paper dicts from Semantic Scholar
    query_embedding = np.random.rand(768)  # SPECTER2 embedding (768D)

    input_data = {"papers": papers, "query_embedding": query_embedding}
    finalists = agent.rank_papers(input_data)

    # Composite scoring formula:
    # semantic = cosine_similarity(query_embedding, paper_embedding)
    # authority = log(1 + influential_citations) / log(1 + max_influential_citations)
    # impact = log(1 + total_citations) / log(1 + max_total_citations)
    # recency = (year - 2015) / (2025 - 2015)  # Linear normalization
    # composite_score = 0.4 * semantic + 0.3 * authority + 0.2 * impact + 0.1 * recency
"""

from typing import Dict, List, Any
import numpy as np
from loguru import logger
from datetime import date


class RankingAgent:
    """Agent that ranks Semantic Scholar papers using composite scoring.

    Combines four factors with weights:
    - Semantic similarity (40%): SPECTER2 cosine similarity
    - Authority (30%): Log-normalized influential citations
    - Impact (20%): Log-normalized total citations
    - Recency (10%): Linear normalization 2015-2025

    Returns top 5-10 papers sorted by composite score descending.
    """

    def rank_papers(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank papers using composite scoring algorithm.

        Args:
            input_data: Dict with keys:
                - papers: List[Dict] of Semantic Scholar papers
                - query_embedding: np.ndarray (384D or 768D SPECTER/SPECTER2 embedding)

        Returns:
            List[Dict]: Top 5-10 papers with added score fields:
                - composite_score: float (0.0-1.0)
                - semantic_score: float (0.0-1.0)
                - authority_score: float (0.0-1.0)
                - impact_score: float (0.0-1.0)
                - recency_score: float (0.0-1.0)
                Plus all original paper fields.

        Raises:
            ValueError: If inputs are invalid.
        """
        papers = input_data.get("papers", [])
        query_embedding = input_data.get("query_embedding")

        self._validate_inputs(papers, query_embedding)

        if not papers:
            logger.warning("No papers provided for ranking")
            return []

        # Compute all sub-scores
        semantic_scores = self._compute_semantic_scores(papers, query_embedding)
        authority_scores = self._compute_authority_scores(papers)
        impact_scores = self._compute_impact_scores(papers)
        recency_scores = self._compute_recency_scores(papers)

        # Combine into composite scores
        composite_scores = self._compute_composite_scores(
            semantic_scores, authority_scores, impact_scores, recency_scores
        )

        # Add scores to papers and sort
        scored_papers = []
        for i, paper in enumerate(papers):
            scored_paper = paper.copy()
            scored_paper.update(
                {
                    "composite_score": float(composite_scores[i]),
                    "semantic_score": float(semantic_scores[i]),
                    "authority_score": float(authority_scores[i]),
                    "impact_score": float(impact_scores[i]),
                    "recency_score": float(recency_scores[i]),
                }
            )
            scored_papers.append(scored_paper)

        # Sort by composite score descending and take top 5-10
        scored_papers.sort(key=lambda p: p["composite_score"], reverse=True)
        finalists = scored_papers[:10]  # Take up to 10, but at least 5 if available

        # Log statistics
        logger.debug(
            "Composite scores: min={:.3f}, max={:.3f}, mean={:.3f}",
            float(composite_scores.min()),
            float(composite_scores.max()),
            float(composite_scores.mean()),
        )

        # Log top 3 paper titles
        top_titles = [p.get("title", "Unknown") for p in finalists[:3]]
        logger.info(
            "Top 3 ranked papers: 1. '{}', 2. '{}', 3. '{}'",
            top_titles[0] if len(top_titles) > 0 else "None",
            top_titles[1] if len(top_titles) > 1 else "None",
            top_titles[2] if len(top_titles) > 2 else "None",
        )

        logger.info(
            "Ranking completed: {} papers processed, {} finalists selected",
            len(papers),
            len(finalists),
        )

        return finalists

    def _validate_inputs(self, papers: List[Dict], query_embedding: Any) -> None:
        """Validate input parameters.

        Week 1: query_embedding can be None (will use citation+recency only).
        Week 2+: query_embedding required for full semantic similarity ranking.
        """
        if not isinstance(papers, list):
            raise ValueError("papers must be a list")

        # Week 1: Allow None query_embedding
        if query_embedding is not None:
            if not isinstance(query_embedding, np.ndarray):
                raise ValueError("query_embedding must be a numpy array")

            if query_embedding.shape not in [(384,), (768,)]:
                raise ValueError(
                    f"query_embedding must be 384 or 768-dimensional, got {query_embedding.shape}"
                )

            if not np.isfinite(query_embedding).all():
                raise ValueError("query_embedding contains non-finite values")

    def _compute_semantic_scores(
        self, papers: List[Dict], query_embedding: np.ndarray | None
    ) -> np.ndarray:
        """Compute semantic similarity scores using cosine similarity.

        Week 1: If query_embedding is None, returns 0.5 for all papers (neutral score).
        Week 2+: Uses actual cosine similarity with query embedding.
        """
        # Week 1: No query embedding, return neutral scores
        if query_embedding is None:
            return np.full(len(papers), 0.5, dtype=np.float32)

        embeddings = []
        valid_indices = []

        for i, paper in enumerate(papers):
            emb_dict = paper.get("embedding")
            if emb_dict and isinstance(emb_dict, dict):
                emb = emb_dict.get("specter_v2")
                if emb is not None:
                    try:
                        emb_array = np.array(emb, dtype=np.float32)
                        # Accept both 384 (older SPECTER) and 768 (SPECTER2) dimensions
                        if (
                            emb_array.shape in [(384,), (768,)]
                            and np.isfinite(emb_array).all()
                        ):
                            embeddings.append(emb_array)
                            valid_indices.append(i)
                        else:
                            logger.warning(
                                "Invalid embedding for paper {}: wrong shape or non-finite",
                                paper.get("paperId", i),
                            )
                    except (ValueError, TypeError):
                        logger.warning(
                            "Failed to parse embedding for paper {}",
                            paper.get("paperId", i),
                        )
                else:
                    logger.warning(
                        "Missing specter_v2 embedding for paper {}",
                        paper.get("paperId", i),
                    )
            else:
                logger.warning(
                    "Missing embedding dict for paper {}", paper.get("paperId", i)
                )

        if not embeddings:
            logger.warning("No valid embeddings found, all semantic scores will be 0.0")
            return np.zeros(len(papers))

        # Batch compute similarities
        emb_array = np.stack(embeddings)  # (n_valid, 384)
        similarities = np.dot(emb_array, query_embedding)  # (n_valid,)
        similarities = np.clip(similarities, -1.0, 1.0)  # Ensure valid cosine range

        # Map back to full array
        scores = np.zeros(len(papers))
        scores[valid_indices] = similarities

        return scores

    def _compute_authority_scores(self, papers: List[Dict]) -> np.ndarray:
        """Compute authority scores from influential citations."""
        citations = []
        for paper in papers:
            cit = paper.get("influentialCitationCount", 0)
            if cit is None:
                logger.warning(
                    "Missing influentialCitationCount for paper {}, using 0",
                    paper.get("paperId", "unknown"),
                )
                cit = 0
            elif not isinstance(cit, (int, float)) or cit < 0:
                logger.warning(
                    "Invalid influentialCitationCount {} for paper {}, using 0",
                    cit,
                    paper.get("paperId", "unknown"),
                )
                cit = 0
            citations.append(cit)

        citations = np.array(citations)
        if citations.max() == 0:
            return np.zeros(len(papers))

        # Log normalization: log(1 + cit) / log(1 + max_cit)
        scores = np.log1p(citations) / np.log1p(citations.max())
        return scores

    def _compute_impact_scores(self, papers: List[Dict]) -> np.ndarray:
        """Compute impact scores from total citations."""
        citations = []
        for paper in papers:
            cit = paper.get("citationCount", 0)
            if cit is None:
                logger.warning(
                    "Missing citationCount for paper {}, using 0",
                    paper.get("paperId", "unknown"),
                )
                cit = 0
            elif not isinstance(cit, (int, float)) or cit < 0:
                logger.warning(
                    "Invalid citationCount {} for paper {}, using 0",
                    cit,
                    paper.get("paperId", "unknown"),
                )
                cit = 0
            citations.append(cit)

        citations = np.array(citations)
        if citations.max() == 0:
            return np.zeros(len(papers))

        # Log normalization: log(1 + cit) / log(1 + max_cit)
        scores = np.log1p(citations) / np.log1p(citations.max())
        return scores

    def _compute_recency_scores(self, papers: List[Dict]) -> np.ndarray:
        """Compute recency scores with linear normalization 2015-date.today().year."""
        years = []
        for paper in papers:
            year = paper.get("year")
            if year is None:
                logger.warning(
                    "Missing year for paper {}, using date.today().year (min recency)",
                    paper.get("paperId", "unknown"),
                )
                year = date.today().year
            elif not isinstance(year, (int, float)):
                logger.warning(
                    "Invalid year {} for paper {}, using date.today().year",
                    year,
                    paper.get("paperId", "unknown"),
                )
                year = date.today().year

            else:
                year = float(year)
                if year < 2015 or year > date.today().year:
                    logger.warning(
                        f"Year {year} outside 2015-{date.today().year} range for paper {paper.get('paperId', 'unknown')}, clamping"
                    )
                    year = np.clip(year, 2015, date.today().year)

            years.append(year)

        years = np.array(years)
        # Linear normalization: (year - 2015) / (2025 - 2015)
        scores = (years - 2015) / (date.today().year - 2015)
        return scores

    def _compute_composite_scores(
        self,
        semantic_scores: np.ndarray,
        authority_scores: np.ndarray,
        impact_scores: np.ndarray,
        recency_scores: np.ndarray,
    ) -> np.ndarray:
        """Compute composite scores using weighted combination.

        Formula: 0.4 * semantic + 0.3 * authority + 0.2 * impact + 0.1 * recency

        Args:
            semantic_scores: Normalized semantic similarity scores (0.0-1.0)
            authority_scores: Normalized authority scores (0.0-1.0)
            impact_scores: Normalized impact scores (0.0-1.0)
            recency_scores: Normalized recency scores (0.0-1.0)

        Returns:
            Composite scores array (0.0-1.0)
        """
        return (
            0.4 * semantic_scores
            + 0.3 * authority_scores
            + 0.2 * impact_scores
            + 0.1 * recency_scores
        )


__all__ = ["RankingAgent"]
