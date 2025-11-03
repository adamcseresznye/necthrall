import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from models.state import State, Paper, PDFContent
from agents.acquisition import AcquisitionAgent
import aiohttp


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def agent():
    """Provides a default AcquisitionAgent for tests."""
    return AcquisitionAgent()


@pytest.mark.asyncio
async def test_agent_returns_state_on_no_papers(agent):
    """Tests that the agent returns the original state if there are no papers."""
    initial_state = State(query="test")
    final_state = await agent(initial_state)
    assert final_state is initial_state


@pytest.mark.asyncio
async def test_agent_handles_no_pdf_urls(agent):
    """Tests that the agent returns original state if papers have no PDF URLs."""
    papers = [
        Paper(
            paper_id="openalex:1",
            title="Paper 1",
            authors=[],
            year=2023,
            journal="N/A",
            citation_count=0,
            doi="10.1000/test1",
            abstract="Test abstract",
            pdf_url=None,
            type="article",
        )
    ]
    initial_state = State(query="test", papers_metadata=papers)
    final_state = await agent(initial_state)
    assert final_state is initial_state


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_agent_downloads_and_updates_state(mock_session_class, agent):
    """Tests a successful download and state update."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.read.return_value = b"pdf content"
    mock_response.raise_for_status = MagicMock()

    mock_get_context_manager = MagicMock()
    mock_get_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session.get = MagicMock(return_value=mock_get_context_manager)
    mock_session_class.return_value.__aenter__.return_value = mock_session

    papers = [
        Paper(
            paper_id="openalex:1",
            title="Paper 1",
            authors=[],
            year=2023,
            journal="N/A",
            citation_count=0,
            doi="10.1000/test1",
            abstract="Test abstract",
            pdf_url="http://example.com/1.pdf",
            type="article",
        )
    ]
    initial_state = State(query="test", papers_metadata=papers)

    agent.extractor.extract = MagicMock(
        return_value=PDFContent(
            paper_id="openalex:1",
            raw_text="text",
            page_count=1,
            char_count=4,
            extraction_time=0.1,
        )
    )

    final_state = await agent(initial_state)

    assert final_state.pdf_contents is not None
    assert len(final_state.pdf_contents) == 1
    assert final_state.pdf_contents[0].paper_id == "openalex:1"
    assert final_state.acquisition_metrics.successful_downloads == 1


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_agent_handles_download_failures(mock_session_class, agent):
    """Tests how the agent handles download failures."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientError("Download failed")
    )

    mock_get_context_manager = MagicMock()
    mock_get_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session.get = MagicMock(return_value=mock_get_context_manager)
    mock_session_class.return_value.__aenter__.return_value = mock_session

    papers = [
        Paper(
            paper_id="openalex:1",
            title="Paper 1",
            authors=[],
            year=2023,
            journal="N/A",
            citation_count=0,
            doi="10.1000/test1",
            abstract="Test abstract",
            pdf_url="http://example.com/1.pdf",
            type="article",
        )
    ]
    initial_state = State(query="test", papers_metadata=papers)

    final_state = await agent(initial_state)

    assert final_state.download_failures is not None
    assert len(final_state.download_failures) == 1
    assert final_state.download_failures[0].paper_id == "openalex:1"
    assert final_state.acquisition_metrics.failed_downloads == 1
