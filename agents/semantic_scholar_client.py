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
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient:
    """Async Semantic Scholar client.

    Responsibilities
    - Run three variant queries in parallel (PRIMARY / BROAD / ALTERNATIVE)
    - Deduplicate by `paperId`
    - Filter to papers with `openAccessPdf.url`
    - Request SPECTER2 embeddings via `fields`
    - Use an asyncio.Semaphore(10) to rate-limit concurrent requests

    The client is intentionally small and uses aiohttp for async HTTP calls.
    """

    def __init__(self, api_key: Optional[str] = None, *, rate_limit: int = 10) -> None:
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
                "embedding",
                "authors",
                "venue",
                "externalIds",
            ]

        # Log entry with a short preview so we can diagnose slow queries
        logger.info(f"multi_query_search entry: queries= {[q[:80] for q in queries]}")

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
            logger.info("multi_query_search finished in {:.3f}s", elapsed)

        # Log how many raw hits we received per input query (primary/broad/alternative
        # ordering is expected by the caller). This helps diagnose which variant
        # produced the most results and identify coverage issues early.
        try:
            # Always log the raw_results structure so we can inspect exact return types/values
            logger.debug(f"Semantic Scholar raw_results repr: {raw_results!r}")

            labels = ["primary", "broad", "alternative"]
            for i, res in enumerate(raw_results):
                label = labels[i] if i < len(labels) else f"query_{i}"
                preview = queries[i][:80] if i < len(queries) else ""

                # Successful list of hits
                if isinstance(res, list):
                    logger.info(
                        f"Semantic Scholar raw hits: {label} returned {len(res)} results (query preview: '{preview}')"
                    )
                # Exception returned by asyncio.gather(..., return_exceptions=True)
                elif isinstance(res, Exception):
                    # Use exception-level logging so stack/trace is captured
                    logger.exception(
                        f"Semantic Scholar query {label} failed for preview '{preview}': {res}"
                    )
                else:
                    # Unexpected return type — include repr for diagnosis
                    logger.info(
                        f"Semantic Scholar raw hits: {label} returned unexpected type {type(res)} (value: {res!r}) (query preview: '{preview}')"
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

        # count how many papers actually have a non-empty embedding vector
        embedded = 0
        for p in papers:
            emb_block = p.get("embedding") or {}
            if not emb_block:
                continue
            # Consider any non-empty list/sequence as an embedding vector
            has_vec = False
            for v in emb_block.values():
                if v is None:
                    continue
                try:
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        has_vec = True
                        break
                    # numpy arrays and similar have __len__
                    if hasattr(v, "__len__") and len(v) > 0:
                        has_vec = True
                        break
                except Exception:
                    # defensive: ignore anything that errors when checking len
                    continue
            if has_vec:
                embedded += 1

        logger.info(
            f"multi_query_search returning {len(papers)} papers, {embedded} have embeddings (deduped & filtered)"
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
                    elif resp.status == 503:
                        text = await resp.text()
                        logger.error(
                            f"Semantic Scholar service unavailable (503) for query={query}: {text}"
                        )
                        raise RuntimeError("Semantic Scholar service unavailable (503)")
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
        defaults when fields are missing. Inline comments explain each
        transformation to aid readability.
        """
        # Extract embedding if present. The search API returns embeddings as a nested
        # dict with 'model' and 'vector' keys (e.g., {'model': 'specter_v1', 'vector': [...]}).
        # We normalize this to store the vector under 'specter_v2' key for consistency
        # with downstream pipeline expectations (quality_gate, ranking_agent).
        specter_vec = None
        emb_block = paper.get("embedding")
        if isinstance(emb_block, dict):
            # Check for the nested structure returned by search API
            if "vector" in emb_block:
                specter_vec = emb_block["vector"]
            # Fallback: check for already-normalized specter_v2 key
            elif "specter_v2" in emb_block:
                specter_vec = emb_block["specter_v2"]
        # Handle flattened key format (rare, for backwards compatibility)
        elif "embedding.specter_v1" in paper:
            specter_vec = paper.get("embedding.specter_v1")
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
            # Embedding: normalize to specter_v2 key regardless of source model
            "embedding": {"specter": specter_vec} if specter_vec is not None else {},
            # Authors list and venue
            "authors": paper.get("authors", []),
            "venue": paper.get("venue"),
            # External identifiers like DOI or ArXiv id
            "externalIds": paper.get("externalIds", {}),
        }

        return normalized


__all__ = ["SemanticScholarClient"]
