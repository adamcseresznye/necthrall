import tracemalloc
import asyncio
import pytest

from agents.analysis import AnalysisAgent
from models.state import State

pytestmark = [pytest.mark.integration]


@pytest.mark.asyncio
async def test_analysis_memory_within_limit(
    realistic_scientific_papers, realistic_passages_factory, monkeypatch
):
    """
    Use tracemalloc to ensure AnalysisAgent.analyze stays within 200MB traced allocations for a representative run.

    We monkeypatch LLM calls to avoid external network and keep test deterministic.
    """

    agent = AnalysisAgent()

    # Prevent real LLM calls by stubbing contradiction detector's LLM call
    async def _fake_perform_contradiction(self_state):
        return []

    # Patch the instance's contradiction detector method
    agent.contradiction_detector._perform_contradiction_detection = (
        lambda *args, **kwargs: asyncio.sleep(0) or []
    )

    # Build a representative state
    state = State(original_query="cardio effects")
    state.filtered_papers = realistic_scientific_papers
    state.relevant_passages = realistic_passages_factory(count=15, seed=2)

    tracemalloc.start()
    result = await agent.analyze(state)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert bytes to MB
    peak_mb = peak / 1024 / 1024

    assert peak_mb < 200, f"Peak traced allocations too large: {peak_mb:.2f} MB"
    assert "credibility_scores" in result
