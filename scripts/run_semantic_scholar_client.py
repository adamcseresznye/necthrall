import asyncio
import time
import os
from dotenv import load_dotenv

load_dotenv()

from agents.semantic_scholar_client import SemanticScholarClient

QUERIES = [
    "intermittent fasting cardiovascular",
    "intermittent fasting heart health",
    "time-restricted eating cardiovascular",
]


async def main():
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print("SEMANTIC_SCHOLAR_API_KEY not set; please set it in .env or environment")
        return

    client = SemanticScholarClient(api_key=api_key)

    # Use the same fields the client would request to get consistent counts
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

    start = time.perf_counter()
    import aiohttp

    query_raw_counts = []
    async with aiohttp.ClientSession() as session:
        for q in QUERIES:
            try:
                data = await client._fetch_query(session, q, 100, fields)
                query_raw_counts.append(len(data))
            except Exception as e:
                print(f"Query error for '{q}': {e}")
                query_raw_counts.append(0)

    # Now run the full multi_query_search to get deduped & filtered papers
    papers = await client.multi_query_search(QUERIES, limit_per_query=100)
    elapsed = time.perf_counter() - start

    # Compute stats
    total = len(papers)
    papers_with_pdf = sum(
        1
        for p in papers
        if p.get("openAccessPdf") and p.get("openAccessPdf").get("url")
    )
    papers_with_embedding = sum(
        1 for p in papers if p.get("embedding") and p.get("embedding").get("specter_v2")
    )

    for idx, (q, cnt) in enumerate(zip(QUERIES, query_raw_counts), start=1):
        print(f"Query {idx}: {q} - {cnt} papers")

    print(f"Total after deduplication: {total} papers")
    print(
        f"Papers with PDFs: {papers_with_pdf} ({(papers_with_pdf/total*100) if total else 0:.0f}%)"
    )
    print(
        f"Papers with embeddings: {papers_with_embedding} ({(papers_with_embedding/total*100) if total else 0:.0f}%)"
    )
    print(f"Execution time: {elapsed:.2f}s")

    print(papers[:3])  # Print first 3 papers as sample


if __name__ == "__main__":
    asyncio.run(main())
