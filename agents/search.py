import requests
import time
from loguru import logger
from typing import List, Dict, Any
import json

from utils.openalex_client import (
    OpenAlexClient,
    Paper as DataclassPaper,
    _reconstruct_abstract,
)
from models.state import State, Paper as PydanticPaper
from utils.pipeline_logging import log_pipeline_stage
import json
import os


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

    # --- Diagnostic helpers inserted for API-vs-website debugging ---
    def _debug_api_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a low-level OpenAlex API call and log detailed metadata for diagnosis.

        Returns the parsed JSON or an empty dict on failure.
        """
        # Include mailto when present to mirror website behaviour
        mailto = os.getenv("OPENALEX_EMAIL")
        if mailto:
            params.setdefault("mailto", mailto)

        try:
            logger.info("🔍 OPENALEX API DEBUG: Full request details below")
            logger.info(f"├─ Full URL: {self.OPENALEX_BASE_URL}")
            logger.info(f"├─ Parameters: {json.dumps(params, indent=2)}")

            resp = requests.get(f"{self.OPENALEX_BASE_URL}", params=params, timeout=30)
            status = resp.status_code
            data = resp.json()

            results = data.get("results", [])
            meta = data.get("meta", {})

            logger.info(f"├─ HTTP Status: {status}")
            logger.info(f"├─ Results Returned: {len(results)}")
            logger.info(
                f"├─ Total Available (meta.count): {meta.get('count', 'unknown')}"
            )
            logger.info(
                f"├─ Per Page (meta.per_page): {meta.get('per_page', 'unknown')}"
            )
            logger.info(f"└─ Current Page (meta.page): {meta.get('page', 'unknown')}")

            if results:
                logger.info("📄 First 3 Paper IDs:")
                for i, paper in enumerate(results[:3]):
                    logger.info(
                        f"   {i+1}. {paper.get('id', paper.get('doi', 'no-id'))}"
                    )

            return data

        except Exception as e:
            logger.exception(f"OpenAlex debug API call failed: {e}")
            return {}

    def search_by_type(
        self, query: str, paper_type: str, per_page: int = 25, page: int = 1
    ):
        """Search with detailed parameter logging (diagnostic)."""
        params = {
            "search": query,
            "filter": f"type:{paper_type},is_oa:true",
            "per_page": per_page,
            "page": page,
        }

        return self._debug_api_call(params)

    def debug_filter_combinations(self, query: str) -> None:
        """Test multiple filter combinations and log their returned counts."""
        filter_tests = [
            ("current_articles", f"type:article,is_oa:true"),
            ("current_reviews", f"type:review,is_oa:true"),
            ("combined_types", f"type:article|review,is_oa:true"),
            ("with_pdf_filter", f"type:article|review,is_oa:true,has_pdf:true"),
            ("no_pdf_requirement", f"type:article|review,is_oa:true"),
            ("website_equivalent", f"is_oa:true,has_pdf:true"),
        ]

        logger.info("🧪 FILTER COMBINATION TESTS:")
        total_found = 0

        for test_name, filter_str in filter_tests:
            params = {
                "search": query,
                "filter": filter_str,
                "per_page": 50,
                "page": 1,
            }

            data = self._debug_api_call(params)
            results = data.get("results", []) if data else []
            meta_count = (
                data.get("meta", {}).get("count", "unknown") if data else "unknown"
            )

            logger.info(f"├─ {test_name}: {len(results)} returned, {meta_count} total")

            if test_name.startswith("current_"):
                total_found += len(results)

        logger.info(f"└─ Current method total (summed pages): {total_found} papers")

    def analyze_website_equivalent_query(self, query: str) -> None:
        """Attempt a few API parameter combinations that mimic website behaviour."""
        website_equivalent_attempts = [
            {"search": query, "filter": "is_oa:true,has_pdf:true", "per_page": 100},
            {"search": query, "filter": "is_oa:true", "per_page": 100},
            {
                "search": query,
                "filter": "type:article|review|preprint,is_oa:true,has_pdf:true",
                "per_page": 100,
            },
            {
                "search": query,
                "filter": "is_oa:true",
                "sort": "relevance_score:desc",
                "per_page": 100,
            },
        ]

        logger.info("🌐 WEBSITE REPLICATION ATTEMPTS:")

        for i, params in enumerate(website_equivalent_attempts, 1):
            data = self._debug_api_call(params)
            results = data.get("results", []) if data else []
            meta = data.get("meta", {}) if data else {}

            logger.info(
                f"├─ Attempt {i}: {len(results)} returned, {meta.get('count', 'unknown')} total"
            )
            logger.info(f"   Filters: {params.get('filter', '<none>')}")

            if meta.get("count", 0) >= 200:
                logger.info("   🎯 CLOSE MATCH! This might be the correct approach")

    def check_date_filtering(self, query: str) -> None:
        """Check if implicit or explicit date filters explain differences."""
        current_year = 2025

        date_tests = [
            ("no_date_filter", "is_oa:true,has_pdf:true"),
            (
                "last_5_years",
                f"is_oa:true,has_pdf:true,publication_year:>{current_year-5}",
            ),
            (
                "last_10_years",
                f"is_oa:true,has_pdf:true,publication_year:>{current_year-10}",
            ),
            (
                "recent_only",
                f"is_oa:true,has_pdf:true,publication_year:>{current_year-3}",
            ),
        ]

        logger.info("📅 DATE FILTERING ANALYSIS:")

        for test_name, filter_str in date_tests:
            params = {"search": query, "filter": filter_str, "per_page": 100}
            data = self._debug_api_call(params)
            total = data.get("meta", {}).get("count", 0) if data else 0
            logger.info(f"├─ {test_name}: {total} total papers")

    def search(self, state: State) -> State:
        """
        Execute OpenAlex search with type filtering.

        Includes fallback: if optimized query returns <10 papers, retry with original query.
        This prevents over-optimization from narrowing results too much.

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

        # Fallback: if optimized query returned too few papers, retry with original query
        if (
            len(all_papers) < 10
            and state.optimized_query != state.original_query
            and state.original_query
        ):
            logger.warning(
                f"SearchAgent: Optimized query returned only {len(all_papers)} papers. "
                f"Retrying with original query: '{state.original_query}'"
            )
            time.sleep(self.RATE_LIMIT_DELAY)
            review_papers = self._query_openalex(
                state.original_query, paper_type="review", limit=30
            )
            time.sleep(self.RATE_LIMIT_DELAY)
            article_papers = self._query_openalex(
                state.original_query, paper_type="article", limit=70
            )
            all_papers = review_papers + article_papers
            query = (
                state.original_query
            )  # Update query for quality assessment and logging

        # Assess search quality
        search_quality = self._assess_quality(all_papers, query)

        # Convert dataclass Papers to Pydantic Papers for State model
        pydantic_papers = [
            self._convert_to_pydantic_paper(paper) for paper in all_papers
        ]

        # Update state
        state.papers_metadata = pydantic_papers
        state.search_quality = search_quality

        # Record stage counts for pipeline summary / diagnostics
        if not state.processing_stats:
            state.processing_stats = {}
        # number of papers available from OpenAlex before downstream filtering
        state.processing_stats["openalex_available"] = len(all_papers)
        state.processing_stats["search_agent"] = len(pydantic_papers)

        # Log pipeline stage
        try:
            log_pipeline_stage(
                "SearchAgent",
                state.processing_stats.get("openalex_available", 0),
                state.processing_stats.get("search_agent", 0),
            )
        except Exception:
            logger.exception("SearchAgent: Failed to emit pipeline stage log")

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
        # Use paginated requests directly to avoid forcing `has_fulltext` or
        # `has_pdf` filters that can be rejected by the API. Collect up to
        # `limit` papers for the requested paper_type using is_oa:true.
        per_page = 100
        collected: List[DataclassPaper] = []
        page = 1
        base_url = self.OPENALEX_BASE_URL
        mailto = os.getenv("OPENALEX_EMAIL")

        select_fields = (
            "id,title,authorships,publication_year,primary_location,cited_by_count,"
            "best_oa_location,abstract_inverted_index,doi,type"
        )

        # Use relevance-first multi-sort by default for website-equivalent ordering
        # Relevance first, then citation count, then publication year (recent first)
        multi_sort = "relevance_score:desc,cited_by_count:desc,publication_year:desc"

        while len(collected) < limit:
            params = {
                "search": query,
                "filter": f"type:{paper_type},is_oa:true",
                "per_page": per_page,
                "sort": multi_sort,
                "page": page,
                "select": select_fields,
            }
            if mailto:
                params["mailto"] = mailto

            try:
                resp = requests.get(base_url, params=params, timeout=30)
                # If API returns 403 for some filter combos, fall back to looser filter
                if resp.status_code == 403:
                    # try without type restriction (broad OA) as a last resort
                    params["filter"] = "is_oa:true"
                    resp = requests.get(base_url, params=params, timeout=30)

                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])

                for item in results:
                    # reconstruct paper dataclass similar to OpenAlexClient
                    best_oa_location = item.get("best_oa_location")
                    pdf_url = (
                        best_oa_location.get("pdf_url") if best_oa_location else None
                    )
                    authors = [
                        (author.get("author") or {}).get("display_name")
                        for author in (item.get("authorships") or [])
                        if author and author.get("author")
                    ]
                    journal_location = item.get("primary_location")
                    journal = (
                        journal_location.get("source", {}).get("display_name")
                        if journal_location and journal_location.get("source")
                        else None
                    )

                    paper = DataclassPaper(
                        paper_id=item.get("id"),
                        title=item.get("title"),
                        authors=[a for a in authors if a],
                        year=item.get("publication_year"),
                        journal=journal,
                        doi=(
                            item.get("doi", "").replace("https://doi.org/", "")
                            if item.get("doi")
                            else None
                        ),
                        type=item.get("type", paper_type),
                        abstract=_reconstruct_abstract(
                            item.get("abstract_inverted_index")
                        ),
                        pdf_url=pdf_url,
                        citation_count=item.get("cited_by_count", 0),
                    )
                    collected.append(paper)
                    if len(collected) >= limit:
                        break

                meta = data.get("meta", {})
                total_available = meta.get("count", None)
                # stop if no more pages
                if not results or (
                    total_available is not None and page * per_page >= total_available
                ):
                    break

                page += 1
                time.sleep(self.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.warning(f"OpenAlex paginated query failed on page {page}: {e}")
                break

        return collected

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

        Quality Check:
        - For OR queries (broad search): PASS if len(papers) >= 30
        - For other queries: PASS if len(papers) >= 10 AND avg_relevance >= 0.4

        Rationale: OR queries cast a wide net with many terms, so keyword overlap
        scores are naturally low even for relevant papers. For these broad searches,
        rely solely on paper count.

        Args:
            papers: List of retrieved papers
            query: Search query for relevance calculation

        Returns:
            Dict with keys: passed (bool), reason (str), paper_count (int), avg_relevance (float)
        """
        paper_count = len(papers)
        is_or_query = " OR " in query.upper()

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

        # Quality assessment: OR queries skip relevance check
        if is_or_query:
            passed = paper_count >= 30
            reason = (
                f"Found {paper_count} papers (OR query, relevance check skipped)"
                if passed
                else f"Insufficient results for OR query: {paper_count} papers (need 30+)"
            )
        else:
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
