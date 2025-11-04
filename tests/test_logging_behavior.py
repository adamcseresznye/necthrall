import asyncio
import pytest
from loguru import logger

from agents.analysis import AnalysisCredibilityScorer, ContradictionDetector

pytestmark = [pytest.mark.integration]


def test_credibility_scoring_emits_debug_log():
    records = []

    def _sink(msg):
        try:
            records.append(str(msg))
        except Exception:
            records.append(repr(msg))

    # Capture debug logs emitted by the scorer
    logger.add(_sink, level="DEBUG", format="{message}")

    meta = {
        "paper_id": "openalex:1",
        "citation_count": 120,
        "year": 2023,
        "journal": "Nature",
    }
    score = AnalysisCredibilityScorer.score_paper(meta)

    # Ensure a debug log entry about credibility_score was emitted
    assert any(
        '"event": "credibility_score"' in r or "credibility_score" in r for r in records
    )
    assert score.score >= 0 and score.score <= 100


def test_contradiction_formatting_logs_info_and_exposes_extra_fields(
    realistic_passages_factory,
):
    records = []

    def _sink(msg):
        try:
            records.append(str(msg))
        except Exception:
            records.append(repr(msg))

    # Use a format that includes extra structured fields
    logger.add(_sink, level="INFO", format="{message} | {extra}")

    passages = realistic_passages_factory(count=3, seed=1)
    detector = ContradictionDetector()

    # Call _format_passages (synchronous) to produce logs
    formatted = detector._format_passages(
        [
            {
                "paper_id": p.paper_id,
                "text": p.content,
                "paper_title": getattr(p, "paper_title", "Title"),
            }
            for p in passages
        ]
    )

    assert formatted
    # verify at least one info log with passages_formatted exists in captured sink
    assert any(
        "passages_formatted" in r or "Passages formatted for LLM" in r for r in records
    )


@pytest.mark.asyncio
async def test_non_retryable_llm_error_logs_error(monkeypatch):
    records = []

    def _sink(msg):
        try:
            records.append(str(msg))
        except Exception:
            records.append(repr(msg))

    logger.add(_sink, level="WARNING", format="{message} | {extra}")

    detector = ContradictionDetector()

    class MockLLM:
        async def ainvoke(self, messages):
            from google.api_core.exceptions import InvalidArgument

            raise InvalidArgument("401 Unauthorized")

    # Call internal method and ensure it logs a non-retryable LLM error and raises
    with pytest.raises(Exception):
        await detector._call_single_llm(
            MockLLM(), [{"role": "user", "content": "hi"}], "gemini"
        )

    assert any("Non-retryable LLM error" in r or "Non-retryable" in r for r in records)
