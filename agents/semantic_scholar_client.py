"""Async Semantic Scholar client used by the retrieval agent.

Implements search with basic rate limiting, retries and normalization
helper for the pipeline `State`.

Usage example:
    client = SemanticScholarClient(api_key="...")
    papers = await client.multi_query_search(["research query"], limit_per_query=100)

The returned items are normalized dictionaries matching the project's
State.papers expectations.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    """Async Semantic Scholar client.

    Responsibilities
    - Run search queries in parallel
    - Deduplicate by `paperId`
    - Filter to papers with `openAccessPdf.url`
    - Use an asyncio.Semaphore(10) to rate-limit concurrent requests

    The client is intentionally small and uses aiohttp for async HTTP calls.
    """

    def __init__(self, api_key: Optional[str] = None, *, rate_limit: int = 10) -> None:
        self.api_key = api_key
        # Global semaphore to cap concurrent outbound requests
        self._semaphore = asyncio.Semaphore(rate_limit)
        # Default per-request timeout (seconds) used for client sessions
        self._timeout_seconds = 30
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Create persistent session with connection pooling
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=25)
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def multi_query_search(
        self,
        queries: List[str],
        limit_per_query: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple search queries in parallel and return normalized papers.

        Args:
            queries: List[str] of three query variants (primary, broad, alt).
            limit_per_query: number of results to request per query (default 100).
            fields: optional list of fields to request from the API.

        Returns:
            List of normalized paper dicts (deduplicated, filtered).

        Notes:
            - Uses asyncio.gather(..., return_exceptions=True) so a single
              failing query won't cancel the others.
        """
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "citationCount",
                "influentialCitationCount",
                "openAccessPdf",
                "authors",
                "venue",
                "externalIds",
            ]

        # Log entry with a short preview so we can diagnose slow queries
        logger.info(f"multi_query_search entry: queries= {[q[:80] for q in queries]}")

        session = await self._get_session()
        # tasks = [self._run_query(session, q, limit_per_query, fields) for q in queries]
        # Stagger the requests by 0.2 seconds to prevent burst rate-limiting
        tasks = []
        for q in queries:
            tasks.append(self._run_query(session, q, limit_per_query, fields))
            await asyncio.sleep(0.2)

        start = time.perf_counter()
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - start
        logger.info("multi_query_search finished in {:.3f}s", elapsed)

        # Log how many raw hits we received per input query.
        # This helps diagnose which query produced the most results.
        try:
            logger.debug(f"Semantic Scholar raw_results repr: {raw_results!r}")

            for i, res in enumerate(raw_results):
                preview = queries[i][:80] if i < len(queries) else ""

                if isinstance(res, list):
                    logger.info(
                        "Semantic Scholar query[{}] returned {} results ('{}')",
                        i,
                        len(res),
                        preview,
                    )
                elif isinstance(res, Exception):
                    logger.exception(
                        "Semantic Scholar query[{}] failed ('{}'): {}", i, preview, res
                    )
                else:
                    logger.info(
                        "Semantic Scholar query[{}] returned unexpected type {} (value: {!r}) ('{}')",
                        i,
                        type(res),
                        res,
                        preview,
                    )
        except Exception:
            # Non-fatal logging helper; do not fail the search flow if logging errors occur
            logger.exception("Failed to log per-query hit counts")

        # Collect successful results, log exceptions
        hits: List[Dict[str, Any]] = []
        for idx, r in enumerate(raw_results):
            if isinstance(r, Exception):
                # Log with query context and exception stack
                logger.exception(f"Query failed: {queries[idx]}")
            elif isinstance(r, list):
                hits.extend(r)
            else:
                # Unexpected return but try to continue
                logger.warning("Unexpected result type from query: %s", type(r))

        # Deduplicate by paperId AND Normalized Title
        seen_ids = set()
        seen_titles = set()
        papers = []

        for p in hits:
            pid = p.get("paperId")

            # 1. Skip invalid IDs
            if not pid:
                continue

            # 2. Skip if we've seen this ID
            if pid in seen_ids:
                continue

            # 3. Skip if we've seen this TITLE (Normalization: lowercase, first 50 chars)
            # This catches "The Study of X" vs "The study of x" vs "The Study of X (Draft)"
            raw_title = p.get("title", "")
            if not raw_title:
                continue

            norm_title = raw_title.lower().strip()[:50]
            if norm_title in seen_titles:
                continue

            # 4. Filter: require openAccessPdf.url
            oa = p.get("openAccessPdf")
            if not oa or not oa.get("url"):
                continue

            # Mark as seen and add to results
            seen_ids.add(pid)
            seen_titles.add(norm_title)

            # Normalize and append
            papers.append(self.normalize_paper(p))

        logger.info(
            f"multi_query_search returning {len(papers)} papers (deduped & filtered)"
        )
        return papers

    async def _run_query(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int,
        fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Wrapper around _fetch_query that handles retries and backoff."""
        max_retries = 3
        backoff = 2
        for attempt in range(1, max_retries + 1):
            try:
                return await self._fetch_query(session, query, limit, fields)
            except Exception as exc:
                # Log and retry with exponential backoff on transient errors
                logger.exception(
                    f"Query attempt {attempt}/{max_retries} failed for query={query}",
                )
                if attempt == max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

        return []

    async def _fetch_query(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int,
        fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Perform a single search request and return the raw data list.

        This method uses the shared semaphore to limit concurrency.
        """
        params = {
            "query": query,
            "limit": str(limit),
            "fields": ",".join(fields),
            "sort": "relevance",
            "year": "1990-",
            "openAccessPdf": "",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # Semaphore prevents >100 simultaneous requests
        await self._semaphore.acquire()
        try:
            try:
                async with session.get(
                    SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=headers
                ) as resp:
                    # Handle common HTTP responses
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                        except Exception as e:
                            # Malformed JSON or unexpected body
                            logger.exception("Failed to parse JSON for query=%s", query)
                            raise RuntimeError("Malformed JSON response") from e
                        # API returns {"data": [...], ...}
                        return data.get("data", [])
                    elif resp.status == 429:
                        # Rate limited: respect Retry-After header when available
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = float(retry_after)
                                logger.info(f"Received Retry-After={wait}, sleeping")
                                await asyncio.sleep(wait)
                            except Exception:
                                pass
                        text = await resp.text()
                        logger.error(
                            f"Semantic Scholar rate limited (429) for {query}: {text}"
                        )
                        raise RuntimeError("Semantic Scholar API rate limited (429)")
                    elif resp.status in (500, 502, 503, 504):
                        text = await resp.text()
                        logger.error(
                            f"Semantic Scholar transient error ({resp.status}) for query={query}: {text}"
                        )
                        raise RuntimeError(
                            f"Semantic Scholar transient error ({resp.status})"
                        )
                    else:
                        text = await resp.text()
                        logger.error(
                            f"Semantic Scholar returned status {resp.status} for query={query}: {text}"
                        )
                        return []
            except asyncio.TimeoutError as te:
                logger.exception(
                    f"Timeout when calling Semantic Scholar for query={query}"
                )
                raise
            except aiohttp.ClientError as ce:
                logger.exception(
                    f"Network error when calling Semantic Scholar for query={query}"
                )
                raise
        finally:
            self._semaphore.release()

    def normalize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw paper dict from Semantic Scholar into the pipeline schema.

        This function extracts the expected fields and applies sensible
        defaults when fields are missing.
        """
        # Build normalized dict with defaults for missing fields
        normalized: Dict[str, Any] = {
            # Identity
            "paperId": paper.get("paperId"),
            # Textual metadata
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "year": paper.get("year"),
            # Citation statistics with safe defaults
            "citationCount": paper.get("citationCount", 0),
            "influentialCitationCount": paper.get("influentialCitationCount", 0),
            # PDF info: ensure we always return a dict (may be empty)
            "openAccessPdf": paper.get("openAccessPdf") or {},
            # Authors list and venue
            "authors": paper.get("authors", []),
            "venue": paper.get("venue"),
            # External identifiers like DOI or ArXiv id
            "externalIds": paper.get("externalIds", {}),
        }

        return normalized


__all__ = ["SemanticScholarClient"]
