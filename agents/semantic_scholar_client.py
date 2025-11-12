"""Async Semantic Scholar client used by the retrieval agent.

Implements multi-query parallel search with basic rate limiting, retries
and normalization helper for the pipeline `State`.

Usage example:
    client = SemanticScholarClient(api_key="...")
    papers = await client.multi_query_search([
        "primary query",
        "broad query",
        "alternative query",
    ], limit_per_query=100)

The returned items are normalized dictionaries matching the project's
State.papers expectations.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    """Async Semantic Scholar client.

    Responsibilities
    - Run three variant queries in parallel (PRIMARY / BROAD / ALTERNATIVE)
    - Deduplicate by `paperId`
    - Filter to papers with `openAccessPdf.url`
    - Request SPECTER2 embeddings via `fields`
    - Use an asyncio.Semaphore(100) to rate-limit concurrent requests

    The client is intentionally small and uses aiohttp for async HTTP calls.
    """

    def __init__(self, api_key: Optional[str] = None, *, rate_limit: int = 100) -> None:
        self.api_key = api_key
        # Global semaphore to cap concurrent outbound requests
        self._semaphore = asyncio.Semaphore(rate_limit)
        # Default per-request timeout (seconds) used for client sessions
        self._timeout_seconds = 10

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
                "embedding.specter_v2",
                "authors",
                "venue",
                "externalIds",
            ]

        # Log entry with a short preview so we can diagnose slow queries
        logger.debug("multi_query_search entry: queries=%s", [q[:80] for q in queries])

        # Use a pooled connector and a client timeout for better performance
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=25)
        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            tasks = [
                self._run_query(session, q, limit_per_query, fields) for q in queries
            ]

            start = time.perf_counter()
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - start
            logger.debug("multi_query_search finished in {:.3f}s", elapsed)

        # Collect successful results, log exceptions
        hits: List[Dict[str, Any]] = []
        for idx, r in enumerate(raw_results):
            if isinstance(r, Exception):
                # Log with query context and exception stack
                logger.exception("Query failed: %s", queries[idx])
            elif isinstance(r, list):
                hits.extend(r)
            else:
                # Unexpected return but try to continue
                logger.warning("Unexpected result type from query: %s", type(r))

        # Deduplicate by paperId
        seen: Dict[str, Dict[str, Any]] = {}
        for p in hits:
            pid = p.get("paperId")
            if not pid:
                continue
            if pid in seen:
                continue
            # Filter: require openAccessPdf.url
            oa = p.get("openAccessPdf")
            if not oa or not oa.get("url"):
                continue
            seen[pid] = self.normalize_paper(p)

        papers = list(seen.values())

        # Fetch embeddings for all papers using batch paper details endpoint
        if papers:
            logger.debug("Fetching embeddings for %d papers", len(papers))
            papers = await self._enrich_with_embeddings(papers)

        logger.info(
            "multi_query_search returning %d papers (deduped & filtered)", len(papers)
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
        backoff = 0.5
        for attempt in range(1, max_retries + 1):
            try:
                return await self._fetch_query(session, query, limit, fields)
            except Exception as exc:
                # Log and retry with exponential backoff on transient errors
                logger.exception(
                    "Query attempt %d/%d failed for query=%s",
                    attempt,
                    max_retries,
                    query,
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
        params = {"query": query, "limit": str(limit), "fields": ",".join(fields)}
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
                                logger.debug("Received Retry-After=%s, sleeping", wait)
                                await asyncio.sleep(wait)
                            except Exception:
                                pass
                        text = await resp.text()
                        logger.error(
                            "Semantic Scholar rate limited (429) for query=%s: %s",
                            query,
                            text,
                        )
                        raise RuntimeError("Semantic Scholar API rate limited (429)")
                    elif resp.status == 503:
                        text = await resp.text()
                        logger.error(
                            "Semantic Scholar service unavailable (503) for query=%s: %s",
                            query,
                            text,
                        )
                        raise RuntimeError("Semantic Scholar service unavailable (503)")
                    else:
                        text = await resp.text()
                        logger.error(
                            "Semantic Scholar returned status %s for query=%s: %s",
                            resp.status,
                            query,
                            text,
                        )
                        return []
            except asyncio.TimeoutError as te:
                logger.exception(
                    "Timeout when calling Semantic Scholar for query=%s", query
                )
                raise
            except aiohttp.ClientError as ce:
                logger.exception(
                    "Network error when calling Semantic Scholar for query=%s", query
                )
                raise
        finally:
            self._semaphore.release()

    async def _enrich_with_embeddings(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fetch embeddings for papers using the batch paper details endpoint.

        The search API doesn't return embeddings, so we need to make separate
        requests to get them. We use the batch endpoint to fetch up to 500
        papers at a time.
        """
        # Semantic Scholar batch endpoint for paper details
        batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"

        # Process in batches of 500 (API limit)
        batch_size = 500
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=25)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            for i in range(0, len(papers), batch_size):
                batch = papers[i : i + batch_size]
                paper_ids = [p["paperId"] for p in batch]

                # Prepare request body
                params = {"fields": "paperId,embedding"}
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["x-api-key"] = self.api_key

                await self._semaphore.acquire()
                try:
                    async with session.post(
                        batch_url,
                        params=params,
                        headers=headers,
                        json={"ids": paper_ids},
                    ) as resp:
                        if resp.status == 200:
                            batch_data = await resp.json()
                            # batch_data is a list of papers with embeddings
                            embedding_map = {}
                            for paper_detail in batch_data:
                                if paper_detail and "paperId" in paper_detail:
                                    pid = paper_detail["paperId"]
                                    emb = paper_detail.get("embedding")
                                    if emb:
                                        embedding_map[pid] = emb

                            # Update papers with embeddings
                            for paper in batch:
                                pid = paper["paperId"]
                                if pid in embedding_map:
                                    emb_data = embedding_map[pid]
                                    # Extract specter_v2 vector
                                    if (
                                        isinstance(emb_data, dict)
                                        and "vector" in emb_data
                                    ):
                                        paper["embedding"]["specter_v2"] = emb_data[
                                            "vector"
                                        ]
                                    elif isinstance(emb_data, list):
                                        # Sometimes API returns vector directly
                                        paper["embedding"]["specter_v2"] = emb_data
                        else:
                            text = await resp.text()
                            logger.warning(
                                "Failed to fetch embeddings batch (status %d): %s",
                                resp.status,
                                text[:200],
                            )
                except Exception as e:
                    logger.exception("Error fetching embeddings batch: %s", e)
                finally:
                    self._semaphore.release()

        return papers

    def normalize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw paper dict from Semantic Scholar into the pipeline schema.

        This function extracts the expected fields and applies sensible
        defaults when fields are missing. Inline comments explain each
        transformation to aid readability.
        """
        # Extract embedding if present. The API may return either a nested
        # 'embedding' dict containing 'specter_v2' or a flattened key
        # 'embedding.specter_v2'. Prefer nested form when available.
        specter_vec = None
        emb_block = paper.get("embedding")
        if isinstance(emb_block, dict):
            specter_vec = emb_block.get("specter_v2")
        elif "embedding.specter_v2" in paper:
            specter_vec = paper.get("embedding.specter_v2")

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
            # Embedding: only include specter_v2 when available
            "embedding": {"specter_v2": specter_vec} if specter_vec is not None else {},
            # Authors list and venue
            "authors": paper.get("authors", []),
            "venue": paper.get("venue"),
            # External identifiers like DOI or ArXiv id
            "externalIds": paper.get("externalIds", {}),
        }

        return normalized


__all__ = ["SemanticScholarClient"]
