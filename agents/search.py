import requests
import time
import logging
from typing import List, Dict, Any
import json

from utils.openalex_client import OpenAlexClient, Paper as DataclassPaper
from models.state import State, Paper as PydanticPaper

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    SearchAgent retrieves scientific papers from OpenAlex with type filtering.

    Uses two parallel queries:
    - type=review: Retrieves 30 review papers for broad context
    - type=article: Retrieves 70 research articles for specific findings

    Returns 100 papers total for downstream filtering by FilteringAgent.
    """

    OPENALEX_BASE_URL = "https://api.openalex.org/works"
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests (10 req/sec limit)

    def __init__(self):
        self.openalex_client = OpenAlexClient()

    def search(self, state: State) -> State:
        """
        Execute OpenAlex search with type filtering.

        Args:
            state: LangGraph State with optimized_query field

        Returns:
            Updated State with papers_metadata (100 papers) and search_quality assessment
        """
        query = state.optimized_query or state.original_query
        logger.info(f"SearchAgent: Querying OpenAlex for '{query}'")

        # Execute two parallel queries with type filtering
        review_papers = self._query_openalex(query, paper_type="review", limit=30)
        time.sleep(self.RATE_LIMIT_DELAY)
        article_papers = self._query_openalex(query, paper_type="article", limit=70)

        # Combine results
        all_papers = review_papers + article_papers

        # Assess search quality
        search_quality = self._assess_quality(all_papers, query)

        # Convert dataclass Papers to Pydantic Papers for State model
        pydantic_papers = [
            self._convert_to_pydantic_paper(paper) for paper in all_papers
        ]

        # Update state
        state.papers_metadata = pydantic_papers
        state.search_quality = search_quality

        logger.info(
            f"SearchAgent: Retrieved {len(all_papers)} papers ({len(review_papers)} reviews, {len(article_papers)} articles)"
        )
        logger.info(
            f"Search Quality: {search_quality['passed']} - {search_quality['reason']}"
        )

        return state

    def _query_openalex(
        self, query: str, paper_type: str, limit: int
    ) -> List[DataclassPaper]:
        """
        Query OpenAlex API with type filtering.

        Args:
            query: Scientific query string
            paper_type: "review" or "article"
            limit: Number of papers to retrieve

        Returns:
            List of Paper objects with metadata
        """
        return self.openalex_client.search_papers_with_type(query, paper_type, limit)

    def _convert_to_pydantic_paper(
        self, dataclass_paper: DataclassPaper
    ) -> PydanticPaper:
        """
        Convert dataclass Paper to Pydantic Paper for State model.

        Args:
            dataclass_paper: Paper from OpenAlex client (dataclass)

        Returns:
            Pydantic Paper object for State model
        """
        return PydanticPaper(
            paper_id=dataclass_paper.paper_id,
            title=dataclass_paper.title,
            authors=dataclass_paper.authors,
            year=dataclass_paper.year,
            journal=dataclass_paper.journal,
            doi=dataclass_paper.doi,
            citation_count=dataclass_paper.citation_count,
            abstract=dataclass_paper.abstract,
            pdf_url=dataclass_paper.pdf_url,
            type=dataclass_paper.type,
        )

    def _assess_quality(
        self, papers: List[DataclassPaper], query: str
    ) -> Dict[str, Any]:
        """
        Assess search quality to determine if refinement is needed.

        Quality Check (replaces separate node):
        - PASS if: len(papers) >= 10 AND avg_relevance >= 0.4
        - FAIL if: len(papers) < 10 OR avg_relevance < 0.4

        Args:
            papers: List of retrieved papers
            query: Search query for relevance calculation

        Returns:
            Dict with keys: passed (bool), reason (str), paper_count (int), avg_relevance (float)
        """
        paper_count = len(papers)

        # Simple relevance heuristic: keyword overlap between query and titles
        query_terms = set(query.lower().split())
        relevance_scores = []

        for paper in papers:
            title_terms = set(paper.title.lower().split())
            overlap = len(query_terms & title_terms)
            relevance = overlap / len(query_terms) if query_terms else 0
            relevance_scores.append(relevance)

        avg_relevance = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        )

        # Quality assessment
        passed = paper_count >= 10 and avg_relevance >= 0.4
        reason = (
            f"Found {paper_count} papers with avg_relevance {avg_relevance:.2f}"
            if passed
            else f"Insufficient results: {paper_count} papers, avg_relevance {avg_relevance:.2f}"
        )

        return {
            "passed": passed,
            "reason": reason,
            "paper_count": paper_count,
            "avg_relevance": avg_relevance,
        }
