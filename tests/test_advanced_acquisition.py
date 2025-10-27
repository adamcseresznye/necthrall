import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from models.state import State, Paper, PDFContent
from agents.acquisition import AcquisitionAgent, PDFParsingError
import aiohttp


@pytest.mark.asyncio
async def test_integrated_pipeline_and_extraction_failures():
    """
    Tests the full pipeline, including PDF extraction and failure counting.
    """
    agent = AcquisitionAgent()

    papers = [
        Paper(
            paper_id="openalex:good_pdf",
            title="Good PDF",
            authors=[],
            year=2023,
            journal="N/A",
            citation_count=0,
            doi="10.1000/good",
            abstract="Good PDF abstract",
            pdf_url="https://example.com/good.pdf",
            type="article",
        ),
        Paper(
            paper_id="openalex:bad_pdf",
            title="Bad PDF",
            authors=[],
            year=2023,
            journal="N/A",
            citation_count=0,
            doi="10.1000/bad",
            abstract="Bad PDF abstract",
            pdf_url="https://example.com/bad.pdf",
            type="article",
        ),
    ]
    initial_state = State(query="test", papers_metadata=papers)

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()

        # Mock successful downloads for both papers
        mock_response_good = AsyncMock()
        mock_response_good.read.return_value = b"good pdf content"
        mock_response_good.raise_for_status = MagicMock()

        mock_response_bad = AsyncMock()
        mock_response_bad.read.return_value = b"bad pdf content"
        mock_response_bad.raise_for_status = MagicMock()

        mock_get_good = MagicMock()
        mock_get_good.__aenter__ = AsyncMock(return_value=mock_response_good)
        mock_get_good.__aexit__ = AsyncMock(return_value=None)

        mock_get_bad = MagicMock()
        mock_get_bad.__aenter__ = AsyncMock(return_value=mock_response_bad)
        mock_get_bad.__aexit__ = AsyncMock(return_value=None)

        mock_session.get = MagicMock(side_effect=[mock_get_good, mock_get_bad])
        mock_session_class.return_value.__aenter__.return_value = mock_session

        # Mock extraction to fail for one of the PDFs
        def mock_extract(paper_id, content):
            if paper_id == "openalex:bad_pdf":
                raise PDFParsingError("Failed to parse")
            return PDFContent(
                paper_id=paper_id,
                raw_text="text",
                page_count=1,
                char_count=4,
                extraction_time=0.1,
            )

        agent.extractor.extract = mock_extract

        final_state = await agent(initial_state)

        assert len(final_state.pdf_contents) == 1
        assert final_state.pdf_contents[0].paper_id == "openalex:good_pdf"

        assert final_state.acquisition_metrics.successful_downloads == 2
        assert final_state.acquisition_metrics.extraction_failures == 1
