"""
Check what abstract/content is available for papers without PDFs.
"""

import os
import sys
import requests
from dotenv import load_dotenv

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

load_dotenv()

email = os.getenv("OPENALEX_EMAIL")
query = "Environmental fate and toxicological impact of persistent organic pollutants."
url = "https://api.openalex.org/works"

params = {
    "search": query,
    "filter": "type:article,is_oa:true",
    "sort": "publication_date:desc",
    "per_page": 5,
    "select": "id,title,best_oa_location,abstract_inverted_index,publication_year,authorships",
    "mailto": email,
}

response = requests.get(url, params=params, timeout=10)
data = response.json()
results = data.get("results", [])

print(f"Found {len(results)} OA articles\n")
print("=" * 80)

for i, item in enumerate(results, 1):
    title = item.get("title", "N/A")
    year = item.get("publication_year", "N/A")
    best_oa = item.get("best_oa_location")
    pdf_url = best_oa.get("pdf_url") if best_oa else None
    landing = best_oa.get("landing_page_url") if best_oa else None

    abstract_inv = item.get("abstract_inverted_index")
    has_abstract = abstract_inv is not None and len(abstract_inv) > 0

    print(f"\nPaper {i}:")
    print(f"  Title: {title[:100]}")
    print(f"  Year: {year}")
    print(f"  PDF URL: {'YES' if pdf_url else 'NO'}")
    print(f"  Landing page: {'YES' if landing else 'NO'}")
    print(f"  Abstract available: {'YES' if has_abstract else 'NO'}")

    if has_abstract:
        # Reconstruct a snippet
        words = []
        for word, positions in list(abstract_inv.items())[:15]:
            words.append((min(positions), word))
        words.sort()
        snippet = " ".join(w for _, w in words)
        print(f"  Abstract snippet: {snippet}...")

print("\n" + "=" * 80)
print("\nConclusion:")
print("  - Papers without PDFs often HAVE abstracts")
print("  - We should use abstracts when PDFs aren't available")
print("  - This will give us 5 papers instead of 3 for this query")
