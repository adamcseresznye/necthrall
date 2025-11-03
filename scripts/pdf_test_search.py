# Test script for SearchAgent - standalone version without other dependencies
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SearchAgent directly to avoid PyTorch dependencies in other agents
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the SearchAgent class directly
from agents.search import SearchAgent
from models.state import State


def test_search_agent():
    """Test the SearchAgent functionality"""
    print("Testing SearchAgent...")

    # Initialize the search agent
    search_agent = SearchAgent()

    # Create a state object with your query
    state = State(
        original_query="CRISPR gene editing safety",
        optimized_query="CRISPR-Cas9 off-target effects and safety considerations",
        config={"max_papers": 100},  # Maximum papers to return
    )

    # Run the search
    updated_state = search_agent.search(state)

    # Access the results
    papers = updated_state.papers_metadata
    search_quality = updated_state.search_quality

    print(f"Found {len(papers)} papers")
    print(f"Search quality: {search_quality}")

    # Count paper types
    review_count = sum(1 for p in papers if p.type == "review")
    article_count = sum(1 for p in papers if p.type == "article")
    print(f"Reviews: {review_count}, Articles: {article_count}")

    # Iterate through papers
    for i, paper in enumerate(papers[:5]):  # Show first 5 papers
        print(f"\nPaper {i+1}:")
        print(f"  Title: {paper.title}")
        print(f"  Authors: {', '.join(paper.authors)}")
        print(f"  Year: {paper.year}")
        print(f"  Journal: {paper.journal}")
        print(f"  Citations: {paper.citation_count}")
        print(f"  DOI: {paper.doi}")
        print(f"  Type: {paper.type}")
        print(f"  PDF: {paper.pdf_url}")

    print("\n✅ SearchAgent test completed successfully!")
    return True


if __name__ == "__main__":
    test_search_agent()
