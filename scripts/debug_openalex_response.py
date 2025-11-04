"""
Debug OpenAlex API response to see what data is actually being returned.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

email = os.getenv("OPENALEX_EMAIL")
print(f"Using email: {email}\n")

url = "https://api.openalex.org/works"

# Test with different filter combinations
test_configs = [
    ("No filters", {}),
    ("Type only", {"type": "article"}),
    ("Type + OA", {"type": "article", "is_oa": "true"}),
    (
        "Type + OA + fulltext",
        {"type": "article", "is_oa": "true", "has_fulltext": "true"},
    ),
]

for config_name, filter_dict in test_configs:
    filter_str = (
        ",".join([f"{k}:{v}" for k, v in filter_dict.items()]) if filter_dict else None
    )

    params = {
        "search": "air pollution health",
        "sort": "publication_date:desc",
        "per_page": 5,
        "select": "id,title,best_oa_location",
        "mailto": email,
    }
    if filter_str:
        params["filter"] = filter_str

    response = requests.get(url, params=params, timeout=10)
    print(f"\n{config_name}: filter={filter_str or 'none'}")

    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        count = data.get("meta", {}).get("count", 0)
        print(f"  Total matching: {count}, Returned: {len(results)}")

        if results:
            with_pdf = sum(
                1
                for r in results
                if r.get("best_oa_location")
                and r.get("best_oa_location").get("pdf_url")
            )
            print(f"  Papers with PDF URL: {with_pdf}/{len(results)}")
    else:
        print(f"  Error {response.status_code}")

# Now test the original query
print("\n" + "=" * 60)
print("Original filter configuration:")
params = {
    "search": "air pollution health",
    "filter": "type:article,has_fulltext:true,is_oa:true",
    "sort": "publication_date:desc",
    "per_page": 5,
    "select": "id,title,authorships,publication_year,primary_location,cited_by_count,best_oa_location,abstract_inverted_index,doi,type",
    "mailto": email,
}

response = requests.get(url, params=params, timeout=10)
print(f"Status: {response.status_code}\n")

if response.status_code == 200:
    data = response.json()
    results = data.get("results", [])
    print(f"Total results returned: {len(results)}\n")

    for i, item in enumerate(results, 1):
        print(f"Paper {i}:")
        print(f"  Title: {item.get('title', 'N/A')}")
        print(f"  ID: {item.get('id', 'N/A')}")

        best_oa = item.get("best_oa_location")
        print(f"  best_oa_location exists: {best_oa is not None}")
        if best_oa:
            print(f"    pdf_url: {best_oa.get('pdf_url', 'MISSING')}")
            print(f"    landing_page_url: {best_oa.get('landing_page_url', 'N/A')}")
            print(f"    license: {best_oa.get('license', 'N/A')}")
        print()
else:
    print(f"Error: {response.text}")
