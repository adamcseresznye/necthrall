"""
Debug script to understand why OpenAlex returns few papers with PDFs.
Tests different filter combinations and shows what's being filtered out.
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Ensure repo root is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

load_dotenv()

email = os.getenv("OPENALEX_EMAIL")
query = "Environmental fate and toxicological impact of persistent organic pollutants."
url = "https://api.openalex.org/works"

print(f"Query: {query}")
print(f"Email: {email}\n")
print("=" * 80)

# Test different filter combinations
test_configs = [
    ("Type:article, OA, fulltext", "type:article,is_oa:true,has_fulltext:true"),
    ("Type:article, OA only", "type:article,is_oa:true"),
    ("Type:article only", "type:article"),
    ("No filters", None),
]

for config_name, filter_str in test_configs:
    params = {
        "search": query,
        "sort": "publication_date:desc",
        "per_page": 10,
        "select": "id,title,best_oa_location,open_access,primary_location",
        "mailto": email,
    }
    if filter_str:
        params["filter"] = filter_str

    print(f"\n{config_name}")
    print(f"  Filter: {filter_str or 'none'}")

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        total = data.get("meta", {}).get("count", 0)
        results = data.get("results", [])

        print(f"  Total matches: {total:,}")
        print(f"  Results returned: {len(results)}")

        # Analyze PDF availability
        with_pdf = 0
        with_landing = 0
        is_oa_count = 0

        for item in results:
            oa = item.get("open_access", {})
            if oa.get("is_oa"):
                is_oa_count += 1

            best_oa = item.get("best_oa_location")
            if best_oa:
                if best_oa.get("pdf_url"):
                    with_pdf += 1
                if best_oa.get("landing_page_url"):
                    with_landing += 1

        print(f"  Papers marked as OA: {is_oa_count}/{len(results)}")
        print(f"  Papers with PDF URL: {with_pdf}/{len(results)}")
        print(f"  Papers with landing page: {with_landing}/{len(results)}")

        if results and not with_pdf:
            print(f"\n  Sample paper (no PDF):")
            sample = results[0]
            print(f"    Title: {sample.get('title', 'N/A')[:80]}")
            print(f"    ID: {sample.get('id', 'N/A')}")
            best_oa = sample.get("best_oa_location")
            if best_oa:
                print(f"    Landing page: {best_oa.get('landing_page_url', 'N/A')}")
                print(f"    PDF URL: {best_oa.get('pdf_url', 'MISSING')}")
                print(f"    License: {best_oa.get('license', 'N/A')}")

            oa = sample.get("open_access", {})
            print(f"    OA status: {oa.get('oa_status', 'N/A')}")
            print(f"    Is OA: {oa.get('is_oa', False)}")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("\nConclusion:")
print("  - If papers exist but have no PDF URLs, we're filtering them out")
print("  - Consider using abstracts from papers with landing pages")
print("  - Or relax the PDF requirement when few results are available")
