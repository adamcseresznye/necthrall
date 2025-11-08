import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from utils.scihub_fallback import SciHubFallback, SciHubResult
from models.state import PDFContent


class TestSciHubFallback:
    """Unit tests for SciHub fallback functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def scihub_fallback(self, temp_dir):
        """Create SciHubFallback instance for testing."""
        return SciHubFallback(temp_dir=temp_dir, auto_cleanup=False)

    @pytest.fixture
    def sample_papers(self):
        """Sample paper metadata for testing."""
        return [
            {
                "id": "https://openalex.org/W2907958255",
                "doi": "https://doi.org/10.1021/ol9910114",
                "title": "Test Paper 1",
            },
            {
                "id": "https://openalex.org/W2483246824",
                "doi": "10.1038/nature12373",
                "title": "Test Paper 2",
            },
            {
                "id": "https://openalex.org/W3164843286",
                "doi": None,  # No DOI - should be skipped
                "title": "Test Paper 3",
            },
        ]

    def test_extract_doi_with_url_prefix(self, scihub_fallback):
        """Test DOI extraction with URL prefix."""
        paper = {"doi": "https://doi.org/10.1021/ol9910114"}
        doi = scihub_fallback._extract_doi(paper)
        assert doi == "10.1021/ol9910114"

    def test_extract_doi_without_prefix(self, scihub_fallback):
        """Test DOI extraction without URL prefix."""
        paper = {"doi": "10.1021/ol9910114"}
        doi = scihub_fallback._extract_doi(paper)
        assert doi == "10.1021/ol9910114"

    def test_extract_doi_alternative_prefix(self, scihub_fallback):
        """Test DOI extraction with alternative URL prefix."""
        paper = {"doi": "http://dx.doi.org/10.1021/ol9910114"}
        doi = scihub_fallback._extract_doi(paper)
        assert doi == "10.1021/ol9910114"

    def test_extract_doi_missing(self, scihub_fallback):
        """Test DOI extraction when DOI is missing."""
        paper = {"doi": None}
        doi = scihub_fallback._extract_doi(paper)
        assert doi is None

        paper = {"doi": ""}
        doi = scihub_fallback._extract_doi(paper)
        assert doi is None

    @patch("subprocess.run")
    def test_ensure_domains_updated_success(self, mock_subprocess, scihub_fallback):
        """Test successful domain update."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = scihub_fallback.ensure_domains_updated()

        assert result is True
        assert scihub_fallback.domains_updated is True
        mock_subprocess.assert_called_once_with(
            ["scidownl", "domain.update", "--mode", "crawl"],
            capture_output=True,
            text=True,
            timeout=60,
        )

    @patch("subprocess.run")
    def test_ensure_domains_updated_failure(self, mock_subprocess, scihub_fallback):
        """Test domain update failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Update failed"
        mock_subprocess.return_value = mock_result

        result = scihub_fallback.ensure_domains_updated()

        assert result is False
        assert scihub_fallback.domains_updated is False
        mock_subprocess.assert_called_once_with(
            ["scidownl", "domain.update", "--mode", "crawl"],
            capture_output=True,
            text=True,
            timeout=60,
        )

    @patch("subprocess.run")
    def test_ensure_domains_updated_cli_fallback(
        self, mock_subprocess, scihub_fallback
    ):
        """Test CLI domain update."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = scihub_fallback.ensure_domains_updated()

        assert result is True
        assert scihub_fallback.domains_updated is True
        mock_subprocess.assert_called_once_with(
            ["scidownl", "domain.update", "--mode", "crawl"],
            capture_output=True,
            text=True,
            timeout=60,
        )

    @patch("subprocess.run")
    def test_ensure_domains_updated_cli_failure(self, mock_subprocess, scihub_fallback):
        """Test CLI domain update failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "CLI error"
        mock_subprocess.return_value = mock_result

        result = scihub_fallback.ensure_domains_updated()

        assert result is False
        assert scihub_fallback.domains_updated is False

    @patch("subprocess.run")
    def test_ensure_domains_updated_only_once(self, mock_subprocess, scihub_fallback):
        """Test that domain update is only attempted once per session."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # First call
        result1 = scihub_fallback.ensure_domains_updated()
        # Second call
        result2 = scihub_fallback.ensure_domains_updated()

        assert result1 is True
        assert result2 is True
        mock_subprocess.assert_called_once()

    def test_download_missing_pdfs_no_papers(self, scihub_fallback):
        """Test download with empty paper list."""
        result = scihub_fallback.download_missing_pdfs([])
        assert result == []

    def test_download_missing_pdfs_max_attempts_limit(
        self, scihub_fallback, sample_papers
    ):
        """Test that all provided papers are processed (no artificial limit)."""
        with patch.object(
            scihub_fallback, "ensure_domains_updated", return_value=True
        ), patch.object(
            scihub_fallback, "_download_extract_cleanup_with_timeout"
        ) as mock_download:

            mock_download.return_value = SciHubResult(
                paper_id="test", doi="test", success=False, error="test"
            )

            # Provide 3 papers - should attempt 2 of them (one has no DOI and gets skipped)
            scihub_fallback.download_missing_pdfs(sample_papers, max_attempts=10)

            # Should attempt papers with DOIs only (2 out of 3)
            assert mock_download.call_count == 2

    @patch("utils.scihub_fallback.scihub_download")
    @patch.object(SciHubFallback, "ensure_domains_updated", return_value=True)
    def test_download_extract_cleanup_success(
        self, mock_domains, mock_scihub_download, scihub_fallback, temp_dir
    ):
        """Test successful download, extract, and cleanup."""

        # Mock scihub_download to create a fake file at the expected path
        def mock_download(**kwargs):
            out_path = kwargs.get("out")
            if out_path:
                fake_pdf_path = Path(out_path)
                fake_pdf_content = b"%PDF-1.4\nFake PDF content for testing"
                fake_pdf_path.write_bytes(fake_pdf_content)
            return None

        mock_scihub_download.side_effect = mock_download

        # Mock text extraction to return content
        with patch.object(
            scihub_fallback,
            "_extract_text_from_file",
            return_value="Extracted text content that is long enough to pass the minimum length requirement for successful extraction and processing",
        ):
            result = scihub_fallback._download_extract_cleanup_with_timeout(
                "W123", "10.1021/test"
            )

            assert result.success is True
            assert (
                result.text_content
                == "Extracted text content that is long enough to pass the minimum length requirement for successful extraction and processing"
            )
            assert result.file_size_bytes > 0

    @patch("utils.scihub_fallback.scihub_download")
    @patch.object(SciHubFallback, "ensure_domains_updated", return_value=True)
    def test_download_extract_cleanup_download_failure(
        self, mock_domains, mock_scihub_download, scihub_fallback
    ):
        """Test download failure handling."""
        mock_scihub_download.side_effect = Exception("Download failed")

        result = scihub_fallback._download_extract_cleanup_with_timeout(
            "W123", "10.1021/test"
        )

        assert result.success is False
        assert "Download exception" in result.error

    @patch("utils.scihub_fallback.scihub_download")
    @patch.object(SciHubFallback, "ensure_domains_updated", return_value=True)
    def test_download_extract_cleanup_extraction_failure(
        self, mock_domains, mock_scihub_download, scihub_fallback, temp_dir
    ):
        """Test text extraction failure handling."""

        # Mock scihub_download to create a fake file
        def mock_download(**kwargs):
            out_path = kwargs.get("out")
            if out_path:
                fake_pdf_path = Path(out_path)
                fake_pdf_content = b"%PDF-1.4\nFake content"
                fake_pdf_path.write_bytes(fake_pdf_content)
            return None

        mock_scihub_download.side_effect = mock_download

        # Mock failed text extraction
        with patch.object(scihub_fallback, "_extract_text_from_file", return_value=""):
            result = scihub_fallback._download_extract_cleanup_with_timeout(
                "W123", "10.1021/test"
            )

            assert result.success is False
            assert "Text extraction failed" in result.error

    def test_extract_text_from_file_success(self, scihub_fallback, temp_dir):
        """Test successful text extraction from PDF file."""
        # Create fake PDF file
        pdf_path = Path(temp_dir) / "test.pdf"
        fake_pdf_content = b"%PDF-1.4\nFake PDF content"
        pdf_path.write_bytes(fake_pdf_content)

        # Mock the PDFTextExtractor.extract to return a PDFContent
        fake_pdfcontent_obj = PDFContent(
            paper_id="W123",
            raw_text="Extracted text",
            page_count=1,
            char_count=len("Extracted text"),
            extraction_time=0.2,
        )
        with patch.object(
            scihub_fallback.pdf_extractor, "extract", return_value=fake_pdfcontent_obj
        ) as mock_extract:
            result = scihub_fallback._extract_text_from_file(str(pdf_path), "W123")

            assert result == "Extracted text"
            mock_extract.assert_called_once()

    def test_extract_text_from_file_failure(self, scihub_fallback, temp_dir):
        """Test text extraction failure handling."""
        # Create fake PDF file
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.write_bytes(b"fake content")

        # Mock extractor to raise exception
        with patch.object(
            scihub_fallback.pdf_extractor,
            "extract",
            side_effect=Exception("Extraction failed"),
        ):
            result = scihub_fallback._extract_text_from_file(str(pdf_path), "W123")

            assert result is None

    def test_cleanup_all_files(self, temp_dir):
        """Test cleanup of temporary directory."""
        scihub_fallback = SciHubFallback(temp_dir=temp_dir, auto_cleanup=False)

        # Create some fake files
        temp_path = Path(temp_dir)
        (temp_path / "file1.pdf").write_bytes(b"fake content 1")
        (temp_path / "file2.pdf").write_bytes(b"fake content 2")

        assert len(list(temp_path.iterdir())) == 2

        scihub_fallback.cleanup_all_files()

        # Directory should be removed
        assert not temp_path.exists()

    @patch("utils.scihub_fallback.scihub_download")
    def test_rate_limiting(self, mock_scihub_download, scihub_fallback):
        """Test rate limiting between downloads."""
        scihub_fallback.min_delay = 0.1  # Short delay for testing
        scihub_fallback.last_download = time.time()

        start_time = time.time()

        # Mock the download to simulate failure
        mock_scihub_download.side_effect = Exception("Download failed")

        # This should trigger rate limiting
        scihub_fallback._download_extract_cleanup_with_timeout("W123", "10.1021/test")

        elapsed = time.time() - start_time
        assert elapsed >= 0.1  # Should have waited at least min_delay

    @patch("utils.scihub_fallback.SCIDOWNL_AVAILABLE", False)
    def test_scidownl_not_available(self, temp_dir):
        """Test behavior when SciDownl is not available."""
        scihub_fallback = SciHubFallback(temp_dir=temp_dir)

        result = scihub_fallback.download_missing_pdfs([{"doi": "10.1021/test"}])
        assert result == []

        result = scihub_fallback.ensure_domains_updated()
        assert result is False


class TestSciHubResult:
    """Test SciHubResult dataclass."""

    def test_scihub_result_creation(self):
        """Test SciHubResult creation with default values."""
        result = SciHubResult(paper_id="W123", doi="10.1021/test", success=False)

        assert result.paper_id == "W123"
        assert result.doi == "10.1021/test"
        assert result.success is False
        assert result.file_path is None
        assert result.text_content is None
        assert result.error is None
        assert result.download_time == 0.0
        assert result.file_size_bytes == 0

    def test_scihub_result_with_values(self):
        """Test SciHubResult creation with all values."""
        result = SciHubResult(
            paper_id="W123",
            doi="10.1021/test",
            success=True,
            file_path="/tmp/test.pdf",
            text_content="Sample text",
            error=None,
            download_time=2.5,
            file_size_bytes=1024,
        )

        assert result.paper_id == "W123"
        assert result.doi == "10.1021/test"
        assert result.success is True
        assert result.file_path == "/tmp/test.pdf"
        assert result.text_content == "Sample text"
        assert result.error is None
        assert result.download_time == 2.5
        assert result.file_size_bytes == 1024


@pytest.mark.integration
class TestSciHubFallbackIntegration:
    """Integration tests that require SciDownl to be installed."""

    @pytest.fixture
    def scihub_fallback_integration(self):
        """Create SciHubFallback for integration testing."""
        return SciHubFallback(auto_cleanup=True)

    def test_domain_update_integration(self, scihub_fallback_integration):
        """Test actual domain update (requires network)."""
        # Skip if SciDownl not available
        if not hasattr(scihub_fallback_integration, "temp_dir"):
            pytest.skip("SciDownl not available")

        result = scihub_fallback_integration.ensure_domains_updated()

        # Should succeed or fail gracefully
        assert isinstance(result, bool)
        assert scihub_fallback_integration.domain_update_attempted is True

    @pytest.mark.slow
    def test_actual_download_integration(self, scihub_fallback_integration):
        """Test actual PDF download (requires network and working SciHub)."""
        # Skip if SciDownl not available
        if not hasattr(scihub_fallback_integration, "temp_dir"):
            pytest.skip("SciDownl not available")

        # Use a well-known open access paper
        sample_paper = {
            "id": "https://openalex.org/W2100837269",
            "doi": "10.1371/journal.pone.0000308",  # PLOS ONE paper (open access)
        }

        result = scihub_fallback_integration.download_missing_pdfs(
            [sample_paper], max_attempts=1
        )

        # Result should be list (may be empty if download fails)
        assert isinstance(result, list)
        assert len(result) <= 1

        if result:
            # If successful, should have expected structure
            pdf_content = result[0]
            assert "paper_id" in pdf_content
            assert "content" in pdf_content
            assert "source" in pdf_content
            assert pdf_content["source"] == "scihub_fallback"
