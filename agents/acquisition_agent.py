from __future__ import annotations

import asyncio
import tempfile
import os
import time
from typing import List, Dict, Any, Optional
from loguru import logger
import aiohttp
from aiohttp import ClientError, TCPConnector

from models.state import State
from utils.pdf_extractor import extract_text_from_pdf_file, PdfExtractionError


class AcquisitionAgent:
    """State-aware async PDF acquisition agent.

    Usage:
        agent = AcquisitionAgent()
        new_state = await agent.process(state)

    Notes:
    - Downloads are performed in parallel (up to 10 concurrent tasks).
    - Each download+extraction is wrapped in a per-PDF timeout (class attr
      `PER_PDF_TIMEOUT`, default 10s).
    - Failures are logged via `logger.warning()` and skipped. Only if *all*
      PDFs fail the State will receive a critical error via `append_error()`.
    """

    PER_PDF_TIMEOUT = 10.0
    _CHUNK_SIZE = 32 * 1024

    async def process(self, state: State) -> State:
        """Process finalists from State, download PDFs in parallel, and
        update `state.passages` with successfully extracted texts.

        Logs progress and timing. Only appends a critical error if zero PDFs
        were acquired.

        Example:
            agent = AcquisitionAgent()
            new_state = await agent.process(state)
        """
        if not state.finalists:
            state.append_error("No finalists available for PDF acquisition")
            return state

        finalists = state.finalists
        total = len(finalists)
        logger.info("Starting PDF acquisition for {n} finalists", n=total)

        start_all = time.monotonic()
        # orchestrate downloads with pooling
        passages = await self._download_pdfs(finalists)
        elapsed = time.monotonic() - start_all

        success = len(passages)
        logger.info(
            "Acquisition complete: {s}/{t} PDFs acquired in {sec:.2f}s",
            s=success,
            t=total,
            sec=elapsed,
        )

        if success == 0:
            state.append_error("Critical: No PDFs acquired")

        state.update_fields(passages=passages)
        return state

    async def _download_pdfs(
        self, finalists: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Download and extract PDFs for a list of finalists in parallel.

        Uses an aiohttp TCPConnector for connection pooling and
        asyncio.gather(return_exceptions=True) to collect results.
        Returns a list of successful passage dicts.
        """
        concurrency = min(10, max(1, len(finalists)))
        connector = TCPConnector(limit_per_host=concurrency)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._process_single_with_semaphore(paper, session)
                for paper in finalists
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # filter successful dicts
        passages = [r for r in results if isinstance(r, dict)]
        return passages

    async def _process_single_with_semaphore(
        self, paper: Dict[str, Any], session: aiohttp.ClientSession
    ) -> Optional[Dict[str, Any]]:
        sem = asyncio.Semaphore(min(10, max(1, 1)))
        async with sem:
            try:
                return await asyncio.wait_for(
                    self._process_single(paper, session), timeout=self.PER_PDF_TIMEOUT
                )
            except asyncio.TimeoutError:
                pid = paper.get("paperId") or paper.get("id") or "unknown"
                logger.warning("Timeout downloading/extracting PDF for {pid}", pid=pid)
            except Exception as e:
                pid = paper.get("paperId") or paper.get("id") or "unknown"
                logger.warning(
                    "Unhandled error for paper {pid}: {err}", pid=pid, err=str(e)
                )
        return None

    async def _process_single(
        self, paper: Dict[str, Any], session: aiohttp.ClientSession
    ) -> Optional[Dict[str, Any]]:
        """High-level single paper processing: download then extract/validate.

        Returns a passage dict on success or None on failure.
        """
        paper_id = paper.get("paperId") or paper.get("id") or "unknown"
        oa = paper.get("openAccessPdf")
        url = oa.get("url") if oa and isinstance(oa, dict) else None

        if not url:
            logger.warning("No PDF URL for paper {pid}; skipping", pid=paper_id)
            return None

        logger.debug("Downloading PDF for {pid} from {url}", pid=paper_id, url=url)
        t0 = time.monotonic()
        try:
            tmp_path = await self._download_single_pdf(paper_id, url, session)
        except asyncio.TimeoutError:
            logger.warning("Timeout while downloading PDF for {pid}", pid=paper_id)
            return None
        except ClientError as e:
            logger.warning(
                "HTTP/client error for {pid}: {err}", pid=paper_id, err=str(e)
            )
            return None
        except Exception as e:
            logger.warning("Download failed for {pid}: {err}", pid=paper_id, err=str(e))
            return None
        dt_download = time.monotonic() - t0

        # extract and validate
        t1 = time.monotonic()
        try:
            text = await self._extract_and_validate(tmp_path, paper_id)
        except PdfExtractionError as e:
            logger.warning(
                "Extraction failed for {pid}: {err}", pid=paper_id, err=str(e)
            )
            return None
        except Exception as e:
            logger.warning(
                "Unexpected extraction error for {pid}: {err}", pid=paper_id, err=str(e)
            )
            return None
        dt_extract = time.monotonic() - t1

        # cleanup temp
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        logger.info(
            "Extracted PDF for {pid} (text_length={n}) download={d:.2f}s extract={e:.2f}s",
            pid=paper_id,
            n=len(text),
            d=dt_download,
            e=dt_extract,
        )

        out = dict(paper)
        out.update(
            {
                "text": text,
                "text_source": "pdf",
                "extraction_error": None,
            }
        )
        return out

    async def _download_single_pdf(
        self, paper_id: str, url: str, session: aiohttp.ClientSession
    ) -> str:
        """Stream-download a single PDF to a temp file and return its path.

        Raises aiohttp.ClientError for network issues or returns if HTTP status
        is non-200 (logged and raises).
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()

        try:
            async with session.get(url) as resp:
                status = resp.status
                if status >= 400:
                    # log specific HTTP errors
                    logger.warning(
                        "HTTP {status} when fetching PDF for {pid}",
                        status=status,
                        pid=paper_id,
                    )
                    raise ClientError(f"HTTP {status}")

                with open(tmp_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(self._CHUNK_SIZE):
                        if not chunk:
                            continue
                        fh.write(chunk)

        except ClientError:
            # ensure file removed on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        return tmp_path

    async def _extract_and_validate(self, tmp_path: str, paper_id: str) -> str:
        """Run PyMuPDF extraction in a thread and validate text length.

        Raises PdfExtractionError on invalid/short extraction.
        """
        text = await asyncio.to_thread(extract_text_from_pdf_file, tmp_path)
        if not text or len(text) < 500:
            raise PdfExtractionError("Extracted text too short (<500 chars)")
        return text


__all__ = ["AcquisitionAgent"]
