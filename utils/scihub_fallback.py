import logging
import tempfile
import os
import subprocess
import atexit
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
import time
import shutil
from unittest.mock import Mock as _Mock

logger = logging.getLogger(__name__)

# Import SciDownl components
try:
    from scidownl import scihub_download

    try:
        # update_link not available in this version, use CLI
        pass
    except:
        pass
    SCIDOWNL_AVAILABLE = True
except ImportError:
    scihub_download = None
    SCIDOWNL_AVAILABLE = False
    logger.warning("scidownl not installed - SciHub fallback disabled")

# Import PDFTextExtractor lazily to avoid circular imports with agents package


@dataclass
class SciHubResult:
    paper_id: str
    doi: str
    success: bool
    file_path: Optional[str] = None
    text_content: Optional[str] = None
    error: Optional[str] = None
    download_time: float = 0.0
    file_size_bytes: int = 0


class SciHubFallback:
    """SciHub fallback with automatic domain updates and immediate cleanup."""

    def __init__(self, temp_dir: Optional[str] = None, auto_cleanup: bool = True):
        if not SCIDOWNL_AVAILABLE:
            logger.warning("SciDownl not available - SciHub fallback disabled")
            # Initialize attributes even if SciDownl is not available
            self.temp_dir = (
                Path(temp_dir)
                if temp_dir
                else Path(tempfile.gettempdir()) / "necthrall_scihub"
            )
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.pdf_extractor = None
            self.auto_cleanup = auto_cleanup
            self.min_delay = 3.0  # 3 seconds between downloads
            self.last_download = 0
            self.domains_updated = False
            self.domain_update_attempted = False
            if auto_cleanup:
                atexit.register(self.cleanup_all_files)

        # Create temp directory
        self.temp_dir = (
            Path(temp_dir)
            if temp_dir
            else Path(tempfile.gettempdir()) / "necthrall_scihub"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import to avoid circular import during test collection
        try:
            from agents.acquisition import PDFTextExtractor

            self.pdf_extractor = PDFTextExtractor()
        except Exception:
            self.pdf_extractor = None
            logger.warning(
                "PDFTextExtractor unavailable - text extraction will be disabled"
            )
        self.auto_cleanup = auto_cleanup

        # More conservative rate limiting for better success
        self.min_delay = 2.0  # 2 seconds between downloads
        self.timeout_per_download = 25  # 25 seconds max per download
        self.last_download = 0

        # Domain update tracking (once per session)
        self.domains_updated = False
        self.domain_update_attempted = False

        # Register cleanup on program exit
        if auto_cleanup:
            atexit.register(self.cleanup_all_files)

    def ensure_domains_updated(self) -> bool:
        """Update SciHub domain list - equivalent to 'scidownl domain.update --mode crawl'"""
        # Allow tests to patch update_link/subprocess even if scidownl not installed;
        # only short-circuit if we've already attempted an update.
        if self.domains_updated or self.domain_update_attempted:
            return self.domains_updated

        # Mark that we've attempted a domain update so tests and callers can
        # observe that the method was invoked even if we short-circuit below.
        # Preserve early-exit behavior if domains already updated or attempted.
        self.domain_update_attempted = True

        # If the module-level SCIDOWNL_AVAILABLE flag is explicitly False,
        # skip attempting a domain update.
        if not SCIDOWNL_AVAILABLE:
            logger.debug("SCIDOWNL_AVAILABLE is False; skipping domain update")
            return False

        logger.info(
            "🌐 Updating SciHub domain list (equivalent to 'scidownl domain.update --mode crawl')..."
        )

        try:
            # Use CLI method
            result = subprocess.run(
                ["scidownl", "domain.update", "--mode", "crawl"],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            if result.returncode == 0:
                self.domains_updated = True
                logger.info("✅ SciHub domains updated via CLI")
            else:
                logger.error(f"❌ CLI domain update failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("❌ Domain update timed out")
        except Exception as e:
            logger.error(f"❌ Domain update error: {e}")

        if not self.domains_updated:
            logger.warning(
                "📝 SciHub domain update failed. You may need to run manually:"
            )
            logger.warning("   scidownl domain.update --mode crawl")

        return self.domains_updated

    def download_missing_pdfs(
        self, missing_papers: List[Dict], max_attempts: int = None
    ) -> List[Dict]:
        """
        Download PDFs with no artificial limits (caller handles selection).
        """
        if not SCIDOWNL_AVAILABLE or not missing_papers:
            return []

        # Use all provided papers (caller already selected appropriate ones)
        papers_to_try = missing_papers

        # Update domains first
        if not self.ensure_domains_updated():
            logger.warning("⚠️ SciHub domains not updated - success rate may be lower")

        logger.info(
            f"🔬 SciHub fallback: Processing {len(papers_to_try)} pre-selected papers"
        )

        pdf_contents = []
        successful_dois = []
        failed_dois = []

        for i, paper in enumerate(papers_to_try):
            paper_id = paper.get("id", f"unknown_{i}")
            doi = self._extract_doi(paper)

            if not doi:
                failed_dois.append(f"{paper_id}: No DOI")
                continue

            logger.info(f"📥 SciHub {i+1}/{len(papers_to_try)}: {doi}")

            # Download with timeout
            start_time = time.time()
            result = self._download_extract_cleanup_with_timeout(paper_id, doi)

            if result.success and result.text_content:
                pdf_content = {
                    "paper_id": paper_id,
                    "content": result.text_content,
                    "source": "scihub_fallback",
                    "extraction_method": "pdf_download",
                }
                pdf_contents.append(pdf_content)
                successful_dois.append(doi)
                logger.info(
                    f"✅ Success: {doi} ({result.file_size_bytes/1024:.1f}KB, {len(result.text_content)} chars)"
                )
            else:
                failed_dois.append(f"{doi}: {result.error}")
                logger.debug(f"❌ Failed: {doi} - {result.error}")

            # Rate limiting
            if i < len(papers_to_try) - 1:
                time.sleep(self.min_delay)

        success_rate = (
            len(pdf_contents) / len(papers_to_try) * 100 if papers_to_try else 0
        )

        logger.info(f"📊 SciHub fallback complete:")
        logger.info(
            f"├─ Successful: {len(pdf_contents)}/{len(papers_to_try)} ({success_rate:.1f}%)"
        )
        logger.info(f"├─ Failed: {len(failed_dois)} papers")
        logger.info(
            f"└─ Success DIIs: {', '.join(successful_dois[:3])}{'...' if len(successful_dois) > 3 else ''}"
        )

        return pdf_contents

    def _download_extract_cleanup_with_timeout(
        self, paper_id: str, doi: str
    ) -> SciHubResult:
        """Download with timeout to prevent hanging."""

        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"SciHub download timeout after {self.timeout_per_download}s"
            )

        # Rate limiting
        elapsed = time.time() - self.last_download
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        start_time = time.time()
        result = SciHubResult(paper_id=paper_id, doi=doi, success=False)
        downloaded_file = None

        try:
            # Set timeout alarm (Unix only - for Windows, we'll use a different approach)
            import signal

            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_per_download)

            # Create filename
            timestamp = int(time.time() * 1000)
            safe_id = (
                paper_id.replace("/", "_").replace("\\", "_").replace(":", "_")[:30]
            )
            output_filename = f"scihub_{safe_id}_{timestamp}.pdf"

            logger.debug(
                f"📥 Downloading {doi} (timeout: {self.timeout_per_download}s)"
            )

            # Download
            scihub_download(keyword=doi, out=str(self.temp_dir / output_filename))
            downloaded_file = str(self.temp_dir / output_filename)

            # Clear timeout alarm
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            result.download_time = time.time() - start_time
            self.last_download = time.time()

            # Process result (same as before)
            if downloaded_file and os.path.exists(downloaded_file):
                result.file_path = downloaded_file
                result.file_size_bytes = os.path.getsize(downloaded_file)
                result.success = True

                # Extract text immediately
                result.text_content = self._extract_text_from_file(
                    downloaded_file, paper_id
                )

                if result.text_content and len(result.text_content.strip()) > 100:
                    logger.debug(
                        f"   ✓ Extracted {len(result.text_content)} chars in {result.download_time:.1f}s"
                    )
                else:
                    result.error = f"Text extraction failed ({len(result.text_content or '')} chars)"
                    result.success = False
            else:
                result.error = "Download failed - no file created"

        except TimeoutError as e:
            result.error = f"Download timeout ({self.timeout_per_download}s)"
            result.download_time = time.time() - start_time
            logger.debug(f"   ⏰ Timeout: {doi}")
        except Exception as e:
            result.error = f"Download exception: {str(e)}"
            result.download_time = time.time() - start_time
            logger.debug(f"   ❌ Exception: {e}")
        finally:
            # Clear timeout alarm
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            self.last_download = time.time()

            # Cleanup file
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.unlink(downloaded_file)
                    logger.debug(f"   🗑️ Cleaned: {os.path.basename(downloaded_file)}")
                except Exception as cleanup_error:
                    logger.warning(f"   ⚠️ Cleanup failed: {cleanup_error}")

        return result

    def _extract_text_from_file(self, file_path: str, paper_id: str) -> Optional[str]:
        """Extract text from PDF file using the project's PDFTextExtractor."""
        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            if not self.pdf_extractor:
                logger.warning("PDFTextExtractor not available; cannot extract text")
                return None

            # Use existing extractor API: extract(paper_id, pdf_bytes) -> PDFContent
            pdf_content = self.pdf_extractor.extract(paper_id, pdf_bytes)
            if pdf_content:
                return pdf_content.raw_text
            return None

        except Exception as e:
            logger.debug(f"Text extraction error: {e}")
            return None

    def _extract_doi(self, paper: Dict) -> Optional[str]:
        """Extract clean DOI from paper metadata."""
        doi = paper.get("doi", "")
        if not doi:
            return None

        # Clean DOI prefixes
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")
        elif doi.startswith("http://dx.doi.org/"):
            doi = doi.replace("http://dx.doi.org/", "")

        return doi.strip() if doi else None

    def cleanup_all_files(self):
        """Clean up temporary directory (called on exit)."""
        if not hasattr(self, "temp_dir") or not self.temp_dir:
            return

        try:
            if self.temp_dir.exists():
                files = list(self.temp_dir.glob("*.pdf"))
                if files:
                    logger.info(
                        f"🗑️ Final cleanup: removing {len(files)} temporary SciHub files"
                    )
                    shutil.rmtree(self.temp_dir)
                    logger.info("✅ SciHub temporary directory cleaned")

        except Exception as e:
            logger.warning(f"⚠️ Final cleanup error: {e}")
