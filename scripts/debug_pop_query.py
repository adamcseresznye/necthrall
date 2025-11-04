"""
Debug OpenAlex to see why we get so few papers for POP query.
Check actual API responses with different filter combinations.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

email = os.getenv("OPENALEX_EMAIL")
query = "Environmental fate and toxicological impact of persistent organic pollutants."

print(f"Query: {query}")
print(f"Email: {email}\n")

url = "https://api.openalex.org/works"

# Test each filter combination that the code tries
test_configs = [
    ("type:review,has_fulltext:true,is_oa:true", "review", 30),
    ("type:review,is_oa:true", "review", 30),
    ("type:review", "review", 30),
    ("type:article,has_fulltext:true,is_oa:true", "article", 70),
    ("type:article,is_oa:true", "article", 70),
    ("type:article", "article", 70),
]

for filter_str, paper_type, per_page in test_configs:
    sort_order = (
        "cited_by_count:desc" if paper_type == "review" else "publication_date:desc"
    )

    params = {
        "search": query,
        "filter": filter_str,
        "sort": sort_order,
        "per_page": per_page,
        "select": "id,title,best_oa_location",
        "mailto": email,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"\nFilter: {filter_str}")
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            total_count = data.get("meta", {}).get("count", 0)

            # Count papers with PDF URLs
            with_pdf = 0
            with_landing = 0
            for r in results:
                oa_loc = r.get("best_oa_location")
                if oa_loc:
                    if oa_loc.get("pdf_url"):
                        with_pdf += 1
                    if oa_loc.get("landing_page_url"):
                        with_landing += 1

            print(f"  Total matching: {total_count:,}")
            print(f"  Returned: {len(results)}")
            print(f"  With PDF URL: {with_pdf}")
            print(f"  With landing page: {with_landing}")

            if results and len(results) > 0:
                sample = results[0]
                print(f"  Sample title: {sample.get('title', 'N/A')[:80]}")
        else:
            print(f"  Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")

print("\n" + "=" * 60)
print("BROADER QUERY TEST")
print("=" * 60)

# Try a simpler, broader query
broad_query = "persistent organic pollutants"
print(f"\nBroader query: {broad_query}")

for paper_type, per_page in [("review", 30), ("article", 70)]:
    filter_str = f"type:{paper_type},is_oa:true"
    sort_order = (
        "cited_by_count:desc" if paper_type == "review" else "publication_date:desc"
    )

    params = {
        "search": broad_query,
        "filter": filter_str,
        "sort": sort_order,
        "per_page": per_page,
        "select": "id,title,best_oa_location",
        "mailto": email,
    }

    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        total_count = data.get("meta", {}).get("count", 0)
        with_pdf = sum(
            1 for r in results if r.get("best_oa_location", {}).get("pdf_url")
        )

        print(f"\n{paper_type.upper()} (is_oa:true only):")
        print(f"  Total matching: {total_count:,}")
        print(f"  Returned: {len(results)}")
        print(f"  With PDF: {with_pdf}")
