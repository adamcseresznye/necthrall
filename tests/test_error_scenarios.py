import asyncio
import pytest
from unittest.mock import AsyncMock

from agents.acquisition import AcquisitionAgent
from models.state import State, Paper, DownloadResult, ErrorReport

from utils.llm_client import LLMClient
from langchain_core.messages import AIMessage
from google.api_core.exceptions import ResourceExhausted, InvalidArgument
from loguru import logger

pytestmark = [pytest.mark.integration]


@pytest.mark.asyncio
async def test_acquisition_network_timeout(monkeypatch):
    """Simulate a network timeout during PDF download and ensure the agent records the failure."""

    async def _fake_download(self, session, paper, semaphore, correlation_id):
        dr = DownloadResult(paper_id=paper.paper_id, success=False)
        err = ErrorReport(
            paper_id=paper.paper_id,
            url=paper.pdf_url,
            error_type="NetworkTimeout",
            message="Simulated timeout",
            timestamp=1234567890.0,
            recoverable=True,
        )
        return dr, err

    monkeypatch.setattr(AcquisitionAgent, "_download_pdf", _fake_download)

    agent = AcquisitionAgent(concurrency_limit=1, timeout=1, max_retries=0)

    state = State(original_query="test query")
    state.papers_metadata = [
        Paper(
            paper_id="openalex:timeout_test",
            title="Timeout Test",
            authors=["T"],
            year=2020,
            journal="Test J",
            citation_count=1,
            pdf_url="http://example.com/fake.pdf",
            type="article",
        )
    ]

    new_state = await agent(state)

    assert new_state.download_failures, "Download failures should be recorded"
    assert new_state.download_failures[0].error_type == "NetworkTimeout"


def test_llm_primary_auth_failure_triggers_fallback(monkeypatch):
    """Primary LLM raises authentication error (InvalidArgument); ensure fallback is used and logs show fallback trigger."""

    # Prepare mock primary that raises InvalidArgument
    class MockPrimary:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            raise InvalidArgument("401 Unauthorized")

    class MockFallback:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return AIMessage(content="Fallback OK")

        def get_num_tokens(self, content):
            return 4

    # Capture logs
    records = []

    def _sink(msg):
        # msg is a Loguru Message object or text depending on version; stringify
        try:
            records.append(str(msg))
        except Exception:
            records.append(repr(msg))

    logger.add(_sink, level="DEBUG", format="{message}")

    monkeypatch.setattr("utils.llm_client.ChatGoogleGenerativeAI", MockPrimary)
    monkeypatch.setattr("utils.llm_client.ChatGroq", MockFallback)

    client = LLMClient()
    resp = client.generate([{"role": "user", "content": "hello"}])

    assert resp["content"] == "Fallback OK"
    # Look for the fallback trigger log
    assert any(
        "llm_fallback_triggered" in r or "llm_call_failure" in r for r in records
    )


def test_llm_primary_rate_limit_logs_rate_limit(monkeypatch):
    """Primary LLM raises ResourceExhausted; ensure 'rate_limit_error' is logged and fallback is used."""

    class MockPrimary:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            raise ResourceExhausted("Rate limit exceeded")

    class MockFallback:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return AIMessage(content="Fallback after rate limit")

        def get_num_tokens(self, content):
            return 2

    records = []

    def _sink(msg):
        try:
            records.append(str(msg))
        except Exception:
            records.append(repr(msg))

    logger.add(_sink, level="DEBUG", format="{message}")

    monkeypatch.setattr("utils.llm_client.ChatGoogleGenerativeAI", MockPrimary)
    monkeypatch.setattr("utils.llm_client.ChatGroq", MockFallback)

    client = LLMClient()
    resp = client.generate([{"role": "user", "content": "hello"}])

    assert resp["content"] == "Fallback after rate limit"
    assert any(
        "rate_limit_error" in r for r in records
    ), "Expected rate_limit_error in logs"
