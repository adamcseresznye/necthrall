"""Collect structured OpenAlex diagnostic stats for a set of queries/filters.

Writes a JSONL file to logs/openalex_stats.jsonl with entries like:
  {"timestamp":..., "test_name":..., "params":..., "http_status":..., "results_returned":..., "meta_count":..., "meta_per_page":..., "meta_page":...}

This is a non-destructive helper for local diagnostics.
"""

import requests
import json
import os
from datetime import datetime

OPENALEX_URL = "https://api.openalex.org/works"


def run_tests(query: str, out_path: str):
    mailto = os.getenv("OPENALEX_EMAIL")

    tests = [
        (
            "current_articles",
            {
                "search": query,
                "filter": "type:article,is_oa:true",
                "per_page": 50,
                "page": 1,
            },
        ),
        (
            "current_reviews",
            {
                "search": query,
                "filter": "type:review,is_oa:true",
                "per_page": 50,
                "page": 1,
            },
        ),
        (
            "combined_types",
            {
                "search": query,
                "filter": "type:article|review,is_oa:true",
                "per_page": 50,
                "page": 1,
            },
        ),
        (
            "no_pdf_requirement",
            {
                "search": query,
                "filter": "type:article|review,is_oa:true",
                "per_page": 50,
                "page": 1,
            },
        ),
        (
            "website_equivalent_pdf",
            {
                "search": query,
                "filter": "is_oa:true,has_pdf:true",
                "per_page": 100,
                "page": 1,
            },
        ),
        (
            "website_equivalent_broad",
            {"search": query, "filter": "is_oa:true", "per_page": 100, "page": 1},
        ),
        (
            "website_equivalent_sort_relevance",
            {
                "search": query,
                "filter": "is_oa:true",
                "sort": "relevance_score:desc",
                "per_page": 100,
                "page": 1,
            },
        ),
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for name, params in tests:
            if mailto:
                params = dict(params)
                params["mailto"] = mailto

            ts = datetime.utcnow().isoformat() + "Z"
            try:
                resp = requests.get(OPENALEX_URL, params=params, timeout=30)
                status = resp.status_code
                try:
                    data = resp.json()
                except Exception:
                    data = {}

                results = data.get("results") if isinstance(data, dict) else None
                results_len = len(results) if isinstance(results, list) else None
                meta = data.get("meta", {}) if isinstance(data, dict) else {}

            except Exception as e:
                status = "error"
                results_len = None
                meta = {}

            record = {
                "timestamp": ts,
                "test_name": name,
                "params": params,
                "http_status": status,
                "results_returned": results_len,
                "meta_count": meta.get("count"),
                "meta_per_page": meta.get("per_page"),
                "meta_page": meta.get("page"),
            }

            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(json.dumps(record))


def main():
    q = (
        "persistent organic pollutants OR POPs OR environmental risk OR ecotoxicity "
        "OR bioaccumulation OR environmental fate OR chemical safety"
    )
    out = os.path.join("logs", "openalex_stats.jsonl")
    run_tests(q, out)


if __name__ == "__main__":
    main()
