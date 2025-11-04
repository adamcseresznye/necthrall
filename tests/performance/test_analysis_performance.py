import time
import pytest
from agents.analysis import AnalysisCredibilityScorer, ContradictionDetector


pytestmark = pytest.mark.performance


def test_credibility_scoring_100_papers_fast():
    # Generate 100 simple paper dicts
    papers = []
    for i in range(100):
        papers.append(
            {
                "paper_id": f"p{i}",
                "citation_count": i * 2,
                "year": 2020 + (i % 5),
                "journal": "Journal",
            }
        )

    start = time.time()
    scores = AnalysisCredibilityScorer.score_papers(papers)
    elapsed = time.time() - start

    assert len(scores) == 100
    # Performance requirement: 100 papers < 1s
    assert elapsed < 1.0, f"Credibility scoring too slow: {elapsed}s"


@pytest.mark.asyncio
async def test_contradiction_detection_latency_with_mock(valid_llm_response_json):
    detector = ContradictionDetector()
    # Patch the llm call to return quickly
    detector._call_llm_with_fallback = pytest.AsyncMock(
        return_value=valid_llm_response_json
    )

    passages = [
        {"paper_id": "p1", "text": "A claims X", "paper_title": "t1"},
        {"paper_id": "p2", "text": "B contradicts X", "paper_title": "t2"},
    ]

    start = time.time()
    results = await detector.detect_contradictions(
        query="q", relevant_passages=passages, llm_config={}
    )
    elapsed = time.time() - start

    assert isinstance(results, list)
    # Expect contradiction detection to be fast when LLM is mocked (<1s)
    assert elapsed < 1.0, f"Contradiction detection too slow: {elapsed}s"
