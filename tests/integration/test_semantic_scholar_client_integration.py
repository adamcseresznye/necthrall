import os
import time
from typing import List

import pytest
from dotenv import load_dotenv

from agents.semantic_scholar_client import SemanticScholarClient

load_dotenv()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_scholar_integration_real_api_key():
    """Integration test against the real Semantic Scholar API.

    Requirements:
        - Requires SEMANTIC_SCHOLAR_API_KEY in the environment (or .env loaded by dev).
        - Performs 3 parallel queries and validates returned papers.

    This test will be skipped if the API key is not available to avoid
    failing CI runs that don't have secrets configured.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        pytest.skip("SEMANTIC_SCHOLAR_API_KEY is not set; skipping integration test")

    client = SemanticScholarClient(api_key=api_key)

    queries: List[str] = [
        "intermittent fasting cardiovascular",
        "intermittent fasting heart health",
        "time-restricted eating cardiovascular outcomes",
    ]

    start = time.perf_counter()
    papers = await client.multi_query_search(queries, limit_per_query=100)
    elapsed = time.perf_counter() - start

    # Timing: network varies; assert it's reasonably fast but allow buffer
    assert elapsed < 12.0, f"Query took too long: {elapsed:.2f}s"

    # Expect several papers after deduplication and PDF filtering
    assert 150 <= len(papers) <= 300, f"Unexpected number of papers: {len(papers)}"

    # Validate a few returned fields are present
    for p in papers[:10]:
        assert p.get("paperId")
        oa = p.get("openAccessPdf")
        assert oa and oa.get("url")

    # If embedding data is returned for a paper, validate its dimensionality
    # (SPECTER2 expected to be 768). Do not fail if no embeddings are present
    # for the returned set — the API does not guarantee embeddings for every
    # paper.
    for p in papers:
        emb = p.get("embedding", {}).get("specter_v2")
        if emb:
            assert isinstance(emb, list)
            assert len(emb) == 768
