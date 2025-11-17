"""Simple helper script to run the SemanticScholarClient and save results.

Usage examples (PowerShell):

  # use API key from env
  $env:SEMANTIC_SCHOLAR_API_KEY = 'your_key_here'
  python semantic_download.py --query "microplastics health" --output results.json

  # pass API key on the command line and append timestamp
  python semantic_download.py --query "microplastics health" --api-key ABC123 --timestamp

The script saves a JSON file with the list of normalized paper dicts returned
by `SemanticScholarClient.multi_query_search`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agents.semantic_scholar_client import SemanticScholarClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def download_results(
    queries, filename: Path, api_key: str | None = None, limit: int = 100
):
    client = SemanticScholarClient(api_key=api_key)
    # Ensure queries is a list
    if isinstance(queries, str):
        queries = [queries]

    logging.info("Running Semantic Scholar searches for %d query(ies)", len(queries))
    papers = await client.multi_query_search(queries, limit_per_query=limit)

    # Write JSON to file
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    logging.info("Saved %d papers to %s", len(papers), filename)
    return len(papers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Semantic Scholar query results to JSON"
    )
    p.add_argument(
        "--query", "-q", required=True, help="Query string (wrap in quotes)."
    )
    p.add_argument(
        "--output", "-o", default="pop_results.json", help="Output JSON filename"
    )
    p.add_argument(
        "--api-key", help="Semantic Scholar API key (overrides env var) if provided"
    )
    p.add_argument(
        "--limit", "-n", type=int, default=100, help="Number of results per query"
    )
    p.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a timestamp to the output filename to avoid clobbering",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        logging.info(
            "No API key provided; if rate limits occur, set SEMANTIC_SCHOLAR_API_KEY env var or use --api-key"
        )

    out_path = Path(args.output)
    if args.timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")

    try:
        api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        limit = 100
        count = asyncio.run(
            download_results(args.query, out_path, api_key=api_key, limit=limit)
        )
        logging.info("Done: saved %d papers to %s", count, out_path)
    except Exception as e:
        logging.exception("Failed to download results: %s", e)


if __name__ == "__main__":
    main()
