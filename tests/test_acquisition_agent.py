import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp
from models.state import State, Paper, PDFContent
from agents.acquisition import AcquisitionAgent


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.asyncio
async def test_agent_handles_mixed_success_and_failure():
    """
    Tests the AcquisitionAgent's ability to handle a mix of successful and failed downloads,
    and to correctly update the state.
    """
    agent = AcquisitionAgent(
        concurrency_limit=10, timeout=10, max_retries=0
    )  # Set retries to 0 for predictable test

    papers_to_download = [
        Paper(
            paper_id="openalex:1",
            title="Paper 1",
            authors=[],
            year=2023,
            journal="arXiv",
            citation_count=0,
            doi="10.48550/arXiv.2303.18223",
            pdf_url="https://arxiv.org/pdf/2303.18223.pdf",
            abstract="",
            type="article",
        ),
        Paper(
            paper_id="openalex:2",
            title="Paper 2",
            authors=[],
            year=2023,
            journal="arXiv",
            citation_count=0,
            doi="10.48550/arXiv.2303.18224",
            pdf_url="https://invalid.url/paper.pdf",
            abstract="",
            type="article",
        ),
    ]

    initial_state = State(query="test", papers_metadata=papers_to_download)

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()

        # Mock successful download response
        mock_response_success = AsyncMock()
        mock_response_success.read.return_value = b"pdf content"
        mock_response_success.raise_for_status = MagicMock()

        # Mock failed download response
        mock_response_failure = AsyncMock()

        # Explicitly assign a synchronous MagicMock to raise_for_status.
        mock_response_failure.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientError("Download failed")
        )

        # Create proper async context manager mocks for each get() call
        mock_get_success = MagicMock()
        mock_get_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_get_success.__aexit__ = AsyncMock(return_value=None)

        mock_get_failure = MagicMock()
        mock_get_failure.__aenter__ = AsyncMock(return_value=mock_response_failure)
        mock_get_failure.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = MagicMock(side_effect=[mock_get_success, mock_get_failure])
        mock_session_class.return_value.__aenter__.return_value = mock_session

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

        # Assertions
        assert len(final_state.pdf_contents) == 1
        assert final_state.pdf_contents[0].paper_id == "openalex:1"
        assert len(final_state.download_failures) == 1
        assert final_state.download_failures[0].paper_id == "openalex:2"
        assert final_state.download_failures[0].error_type == "NetworkError"
        assert final_state.acquisition_metrics.successful_downloads == 1
        assert final_state.acquisition_metrics.failed_downloads == 1
