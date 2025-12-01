"""Manual validation script for SynthesisAgent.

This is a developer tool to verify the Synthesis Agent works correctly
with the real LLM API. It is NOT a CI/CD test.

Usage:
    python scripts/validate_synthesis_manual.py

Prerequisites:
    - Set GOOGLE_API_KEY and/or GROQ_API_KEY in .env
    - Set SYNTHESIS_MODEL and SYNTHESIS_FALLBACK in .env

Expected Output:
    - Query and context passages displayed
    - Synthesized answer with [1], [2], [3] citations
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core.schema import NodeWithScore, TextNode
from agents.synthesis_agent import SynthesisAgent


def create_test_nodes() -> list[NodeWithScore]:
    """Create mock nodes with known content about intermittent fasting."""
    passages = [
        {
            "text": (
                "Intermittent fasting has been shown to promote autophagy, "
                "a cellular cleaning process that removes damaged proteins and "
                "organelles. Studies in mice have demonstrated that fasting for "
                "16-24 hours activates autophagy pathways, potentially reducing "
                "the risk of age-related diseases."
            ),
            "source": "Cell Biology Review, 2023",
        },
        {
            "text": (
                "Research indicates that intermittent fasting can improve insulin "
                "sensitivity by 20-30% in overweight adults. A randomized controlled "
                "trial with 100 participants showed significant reductions in fasting "
                "blood glucose levels after 8 weeks of time-restricted eating."
            ),
            "source": "Diabetes Care Journal, 2022",
        },
        {
            "text": (
                "Cardiovascular benefits of intermittent fasting include reduced "
                "blood pressure, improved cholesterol profiles, and decreased "
                "inflammatory markers. A meta-analysis of 12 studies found that "
                "fasting protocols reduced LDL cholesterol by an average of 10%."
            ),
            "source": "Cardiology Research, 2024",
        },
    ]

    nodes = []
    for i, passage in enumerate(passages):
        node = TextNode(
            text=passage["text"],
            metadata={"source": passage["source"], "index": i + 1},
        )
        nodes.append(NodeWithScore(node=node, score=0.9 - i * 0.1))

    return nodes


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)


async def main() -> None:
    """Run manual validation of SynthesisAgent."""
    print_separator("SYNTHESIS AGENT MANUAL VALIDATION")

    # Create test data
    query = "What are the health benefits of intermittent fasting?"
    nodes = create_test_nodes()

    # Display query
    print_separator("QUERY")
    print(f"  {query}")

    # Display context passages
    print_separator("CONTEXT PASSAGES")
    for i, node_with_score in enumerate(nodes):
        node = node_with_score.node
        source = node.metadata.get("source", "Unknown")
        text_preview = node.text[:100] + "..." if len(node.text) > 100 else node.text
        print(f"\n  [{i + 1}] (score: {node_with_score.score:.2f})")
        print(f"      Source: {source}")
        print(f"      Text: {text_preview}")

    # Initialize agent and synthesize
    print_separator("CALLING SYNTHESIS AGENT")
    print("  Initializing SynthesisAgent...")

    agent = SynthesisAgent(temperature=0.3)

    print("  Calling synthesize()... (waiting for LLM response)")
    print()

    try:
        answer = await agent.synthesize(query=query, nodes=nodes)

        print_separator("SYNTHESIZED ANSWER")
        print()
        print(answer)
        print()

        # Check for citations
        print_separator("CITATION CHECK")
        citations_found = []
        for i in range(1, len(nodes) + 1):
            if f"[{i}]" in answer:
                citations_found.append(i)
                print(f"  ✓ Citation [{i}] found in answer")
            else:
                print(f"  ✗ Citation [{i}] NOT found in answer")

        if len(citations_found) == len(nodes):
            print("\n  ✅ All citations present - synthesis looks correct!")
        elif len(citations_found) > 0:
            print(f"\n  ⚠️  {len(citations_found)}/{len(nodes)} citations found")
        else:
            print("\n  ❌ No citations found - check prompt template")

    except Exception as e:
        print_separator("ERROR")
        print(f"  Synthesis failed: {e}")
        print("\n  Check your .env file for:")
        print("    - GOOGLE_API_KEY or GROQ_API_KEY")
        print("    - SYNTHESIS_MODEL (e.g., gemini/gemini-1.5-flash)")
        print("    - SYNTHESIS_FALLBACK (e.g., groq/llama3-8b-8192)")
        raise

    print_separator("VALIDATION COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
