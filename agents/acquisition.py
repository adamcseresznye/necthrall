import asyncio
import logging
import time
from typing import List, Optional, Dict, Tuple
from collections import defaultdict, Counter
import random
from urllib.parse import urlparse
import json
import os
import uuid
from logging.handlers import RotatingFileHandler

import aiohttp
import fitz
import re
from models.state import (
    Paper,
    PDFContent,
    ErrorReport,
    State,
    AcquisitionMetrics,
    DownloadResult,
)


# --- Custom Exceptions ---
class DownloaderException(Exception):
    """Base exception for downloader errors."""

    def __init__(self, message, paper_id=None, url=None):
        super().__init__(message)
        self.paper_id = paper_id
        self.url = url


class NetworkError(DownloaderException):
    """For network-related errors like timeouts or connection issues."""


class RecoverableError(NetworkError):
    """For recoverable network errors that should trigger a retry."""


class PDFParsingError(DownloaderException):
    """For errors during PDF content parsing or validation."""


class InvalidContentError(DownloaderException):
    """For non-PDF content or other content-related issues."""


class ProgressTracker:
    """Tracks progress and collects metrics for the acquisition process."""

    def __init__(self, total_papers: int):
        self.total_papers = total_papers
        self.successful_downloads = 0
        self.extraction_failures = 0
        self.start_time = time.time()
        self.total_download_time = 0.0
        self.errors: List[ErrorReport] = []

    def record_download_success(self, download_time: float):
        self.successful_downloads += 1
        self.total_download_time += download_time

    def record_failure(self, error: ErrorReport):
        self.errors.append(error)

    def record_extraction_failure(self):
        self.extraction_failures += 1

    def get_metrics(self) -> AcquisitionMetrics:
        total_time = time.time() - self.start_time
        avg_download_time = (
            self.total_download_time / self.successful_downloads
            if self.successful_downloads > 0
            else 0.0
        )
        failure_breakdown = Counter(error.error_type for error in self.errors)

        return AcquisitionMetrics(
            total_papers=self.total_papers,
            successful_downloads=self.successful_downloads,
            failed_downloads=len(self.errors),
            extraction_failures=self.extraction_failures,
            total_time=total_time,
            avg_download_time=avg_download_time,
            failure_breakdown=dict(failure_breakdown),
        )


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON."""

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            log_record["correlation_id"] = record.correlation_id
        if hasattr(record, "paper_id"):
            log_record["paper_id"] = record.paper_id
        if hasattr(record, "url"):
            log_record["url"] = record.url
        if hasattr(record, "error_type"):
            log_record["error_type"] = record.error_type
        return json.dumps(log_record)


def setup_logging():
    """Configures production-ready logging."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = RotatingFileHandler(
        "logs/acquisition.log", maxBytes=10_000_000, backupCount=5
    )
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if os.environ.get("ENV") == "development":
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = setup_logging()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
]


class AcquisitionAgent:
    """
    An agent responsible for concurrently downloading scientific papers and integrating with LangGraph state.
    """

    def __init__(
        self,
        concurrency_limit: int = 10,
        timeout: int = 10,
        max_retries: int = 2,
    ):
        self.concurrency_limit = concurrency_limit
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.extractor = PDFTextExtractor()
        self.circuit_breakers: Dict[str, Dict] = defaultdict(
            lambda: {"failure_count": 0, "is_open": False, "last_failure_time": 0}
        )

    async def __call__(self, state: State) -> State:
        """
        The entry point for the AcquisitionAgent in the LangGraph workflow.
        """
        papers_to_download = state.papers_metadata
        if not papers_to_download or not any(p.pdf_url for p in papers_to_download):
            logger.warning(
                "No papers with PDF URLs to download.",
                extra={"correlation_id": state.request_id},
            )
            return state

        downloads, errors, metrics = await self.run_pipeline(
            papers_to_download, state.request_id
        )

        extracted_content = []
        extraction_failures = 0
        for download in downloads:
            if download.success and download.content:
                try:
                    content = self.extractor.extract(
                        download.paper_id, download.content
                    )
                    if content:
                        extracted_content.append(content)
                except (PDFParsingError, InvalidContentError):
                    extraction_failures += 1

        metrics.extraction_failures = extraction_failures

        # Create a new state object with the results
        new_state = state.model_copy(deep=True)
        new_state.pdf_contents = extracted_content
        new_state.download_failures = errors
        new_state.acquisition_metrics = metrics
        return new_state

    async def run_pipeline(
        self, papers: List[Paper], correlation_id: str
    ) -> Tuple[List[DownloadResult], List[ErrorReport], AcquisitionMetrics]:
        """Runs the full download and extraction pipeline."""
        tracker = ProgressTracker(total_papers=len(papers))
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False)

        async with aiohttp.ClientSession(
            connector=connector, timeout=self.timeout
        ) as session:
            tasks = [
                self._download_pdf(session, paper, semaphore, correlation_id)
                for paper in papers
                if paper.pdf_url
            ]
            results = await asyncio.gather(*tasks)

        downloads = []
        for res, error in results:
            if res.success:
                downloads.append(res)
                tracker.record_download_success(res.download_time)
            else:
                if error:
                    tracker.record_failure(error)

        return downloads, tracker.errors, tracker.get_metrics()

    async def _download_pdf(
        self,
        session: aiohttp.ClientSession,
        paper: Paper,
        semaphore: asyncio.Semaphore,
        correlation_id: str,
    ) -> Tuple[DownloadResult, Optional[ErrorReport]]:
        async with semaphore:
            start_time = time.time()

            try:
                host = urlparse(paper.pdf_url).hostname
                if not host:
                    raise InvalidContentError("Invalid URL format")
            except (ValueError, TypeError):
                error = ErrorReport(
                    paper_id=paper.paper_id,
                    url=paper.pdf_url or "N/A",
                    error_type="InvalidContentError",
                    message="Malformed or invalid URL",
                    timestamp=time.time(),
                    recoverable=False,
                )
                return DownloadResult(paper_id=paper.paper_id, success=False), error

            for attempt in range(self.max_retries + 1):
                headers = {"User-Agent": random.choice(USER_AGENTS)}
                try:
                    async with session.get(paper.pdf_url, headers=headers) as response:
                        response.raise_for_status()
                        content = await response.read()
                        download_time = time.time() - start_time
                        return (
                            DownloadResult(
                                paper_id=paper.paper_id,
                                success=True,
                                content=content,
                                download_time=download_time,
                                file_size=len(content),
                            ),
                            None,
                        )
                except aiohttp.ClientError as e:
                    message = f"Attempt {attempt + 1} failed: {e}"
                    if attempt >= self.max_retries:
                        error_report = ErrorReport(
                            paper_id=paper.paper_id,
                            url=paper.pdf_url,
                            error_type="NetworkError",
                            message=message,
                            timestamp=time.time(),
                            recoverable=False,
                        )
                        return (
                            DownloadResult(paper_id=paper.paper_id, success=False),
                            error_report,
                        )
                    await asyncio.sleep(2**attempt)
                except Exception as e:
                    error_report = ErrorReport(
                        paper_id=paper.paper_id,
                        url=paper.pdf_url,
                        error_type="UnknownError",
                        message=str(e),
                        timestamp=time.time(),
                        recoverable=False,
                    )
                    return (
                        DownloadResult(paper_id=paper.paper_id, success=False),
                        error_report,
                    )

            # This should not be reached
            final_error = ErrorReport(
                paper_id=paper.paper_id,
                url=paper.pdf_url,
                error_type="MaxRetriesReached",
                message="All retry attempts failed.",
                timestamp=time.time(),
                recoverable=False,
            )
            return DownloadResult(paper_id=paper.paper_id, success=False), final_error


class PDFTextExtractor:
    """
    Extracts, cleans, and validates text from PDF byte content.
    """

    def __init__(self, min_text_length: int = 100):
        self.min_text_length = min_text_length

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract(self, paper_id: str, pdf_content: bytes) -> Optional[PDFContent]:
        start_time = time.time()
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
        except RuntimeError as e:
            raise PDFParsingError(f"Failed to open PDF: {e}", paper_id=paper_id)

        with doc:
            if doc.is_encrypted:
                raise InvalidContentError("PDF is encrypted", paper_id=paper_id)

            full_text = "".join(page.get_text() for page in doc)
            cleaned_text = self._clean_text(full_text)

            if len(cleaned_text) < self.min_text_length:
                raise InvalidContentError(
                    "Insufficient text content", paper_id=paper_id
                )

            return PDFContent(
                paper_id=paper_id,
                raw_text=cleaned_text,
                page_count=doc.page_count,
                char_count=len(cleaned_text),
                extraction_time=time.time() - start_time,
            )
