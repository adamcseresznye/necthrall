"""
Direct test of OpenAlex API to diagnose search issues.
"""

import os
import sys
from dotenv import load_dotenv
import requests

# Ensure repo root is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

load_dotenv()

email = os.getenv("OPENALEX_EMAIL")
print(f"Testing OpenAlex API with email: {email}\n")

# Test 1: Simple direct query
test_queries = [
    "air pollution health",
    "persistent organic pollutants",
    "cardiovascular disease",
]

for query in test_queries:
    print(f"Query: '{query}'")

    # Try the API endpoint directly
    url = "https://api.openalex.org/works"
    params = {"filter": f"default.search:{query}", "per_page": 5, "mailto": email}

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            count = data.get("meta", {}).get("count", 0)
            results = data.get("results", [])
            print(f"  Results: {count} total, {len(results)} returned")

            if results:
                print(f"  First result: {results[0].get('title', 'N/A')}")
            else:
                print(f"  Response meta: {data.get('meta', {})}")
        else:
            print(f"  Error: {response.text[:200]}")
    except Exception as e:
        print(f"  Exception: {e}")

    print()

# Test 2: Check if SearchAgent is using the email correctly
print("\n" + "=" * 60)
print("Testing SearchAgent implementation...")
print("=" * 60)

from agents.search import SearchAgent
from models.state import State

agent = SearchAgent()
print(f"SearchAgent openalex_client.mailto: {agent.openalex_client.mailto}")

# Test the client directly first
print("\nTesting OpenAlexClient.search_papers_with_type directly...")
try:
    papers = agent.openalex_client.search_papers_with_type(
        query="air pollution health", paper_type="article", max_results=5
    )
    print(f"  Direct client call returned: {len(papers)} papers")
    if papers:
        print(f"  First paper: {papers[0].title}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback

    traceback.print_exc()

# Test search
test_state = State(
    original_query="air pollution health effects",
    optimized_query="air pollution health effects",
)

print("\nRunning SearchAgent.search()...")
try:
    result_state = agent.search(test_state)
    print(f"Papers returned: {len(result_state.papers_metadata or [])}")

    if result_state.papers_metadata:
        for i, paper in enumerate(result_state.papers_metadata[:3], 1):
            print(f"  {i}. {paper.title}")
    else:
        print("  No papers found")
        print(f"  Search stats: {getattr(result_state, 'search_stats', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
