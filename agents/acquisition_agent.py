from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from curl_cffi.requests import AsyncSession, RequestsError
from loguru import logger

from models.state import State
from utils.pdf_extractor import PdfExtractionError, extract_text_from_pdf_file

# UPDATE THIS BLOCK
HEADERS = {
    # Modern User-Agent (Chrome 120+)
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Mimic coming from Google Scholar (Crucial for bypassing blocks)
    "Referer": "https://scholar.google.com/",
    # Standard browser Accept headers
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
}


class AcquisitionAgent:
    """State-aware async PDF acquisition agent.

    Usage:
        agent = AcquisitionAgent()
        new_state = await agent.process(state)

    Notes:
    - Downloads are performed in parallel (up to 10 concurrent tasks).
    - Each download+extraction is wrapped in a per-PDF timeout (class attr
      `PER_PDF_TIMEOUT`, default 3s).
    - Failures are logged via `logger.warning()` and skipped. Only if *all*
      PDFs fail the State will receive a critical error via `append_error()`.
    """

    PER_PDF_TIMEOUT = 30.0
    _CHUNK_SIZE = 32 * 1024

    async def process(self, state: State) -> State:
        """Process finalists with Base+Bonus strategy:
        - Base: Index ALL abstracts from state.finalists
        - Bonus: Upgrade top 5 accessible papers to full PDF text

        Args:
            state: State containing finalists (List[Dict] of papers)

        Returns:
            Updated state with passages containing abstract + PDF content

        Example:
            agent = AcquisitionAgent()
            new_state = await agent.process(state)
        """
        if not state.finalists:
            state.append_error("No finalists available for acquisition")
            return state

        TARGET_PDF_COUNT = 5
        acquired_pdfs = 0
        passages_map: Dict[str, Dict[str, Any]] = {}

        finalists = state.finalists
        logger.info(
            "Starting Base+Bonus acquisition: {n} abstracts, target {t} PDFs",
            n=len(finalists),
            t=TARGET_PDF_COUNT,
        )

        start_all = time.monotonic()

        async with AsyncSession(impersonate="chrome110", headers=HEADERS) as session:
            # 1. First, add abstracts for ALL finalists (fast, synchronous)
            for paper in finalists:
                paper_id = paper.get("paperId") or paper.get("id")
                if not paper_id:
                    continue

                abstract_text = paper.get("abstract", "")
                if abstract_text:
                    abstract_passage = dict(paper)
                    abstract_passage.update(
                        {
                            "text": abstract_text,
                            "text_source": "abstract",
                            "extraction_error": None,
                        }
                    )
                    passages_map[paper_id] = abstract_passage

            # 2. Concurrent PDF acquisition for top candidates
            # We try the top 12 candidates to get our target of 5 PDFs
            pdf_candidates = [
                (idx, p) for idx, p in enumerate(finalists) if self._has_pdf_url(p)
            ][:12]

            if pdf_candidates:
                logger.info(
                    "Attempting concurrent PDF download for top {n} candidates",
                    n=len(pdf_candidates),
                )

                async def fetch_pdf_safe(idx, paper):
                    try:
                        return await asyncio.wait_for(
                            self._process_single(paper, session),
                            timeout=self.PER_PDF_TIMEOUT,
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.warning(
                            f"Timeout/Cancelled while downloading PDF for paper {idx + 1}"
                        )
                        return None
                    except Exception as e:
                        logger.warning(
                            f"Error downloading PDF for paper {idx + 1}: {e}"
                        )
                        return None

                # Launch all tasks concurrently
                tasks = [fetch_pdf_safe(idx, p) for idx, p in pdf_candidates]
                results = await asyncio.gather(*tasks)

                # Collect successful results up to target
                for res in results:
                    # We can restrict the number of acquired PDFs here
                    # if acquired_pdfs >= TARGET_PDF_COUNT:
                    #    break

                    if res and res.get("text"):
                        paper_id = res.get("paperId") or res.get("id")
                        if paper_id:
                            passages_map[paper_id] = res
                            acquired_pdfs += 1
                            paper_title = res.get("title", "Unknown")[:60]
                            logger.info(
                                "Upgrading to PDF ({a}/{t}): {title}",
                                a=acquired_pdfs,
                                t=TARGET_PDF_COUNT,
                                title=paper_title,
                            )

        elapsed = time.monotonic() - start_all

        # Update state.passages with final results
        final_passages = list(passages_map.values())
        new_state = state.model_copy(deep=True)
        new_state.passages = final_passages

        if not final_passages:
            new_state.append_error("Critical: No PDFs acquired")

        logger.info(
            "Acquisition complete: {total} total passages ({pdfs} PDFs, {abstracts} abstracts) in {sec:.2f}s",
            total=len(final_passages),
            pdfs=acquired_pdfs,
            abstracts=len(final_passages) - acquired_pdfs,
            sec=elapsed,
        )

        return new_state

    def _has_pdf_url(self, paper: Dict[str, Any]) -> bool:
        """Check if paper has a PDF URL available."""
        oa_pdf = paper.get("openAccessPdf")
        return bool(oa_pdf and isinstance(oa_pdf, dict) and oa_pdf.get("url"))

    async def _process_single(
        self, paper: Dict[str, Any], session: AsyncSession
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

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, f"{paper_id}.pdf")
            t0 = time.monotonic()
            try:
                await self._download_single_pdf(
                    paper_id, url, session, destination_path=pdf_path
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout while downloading PDF for {pid}", pid=paper_id)
                return None
            except RequestsError as e:
                logger.warning(
                    "HTTP/client error for {pid}: {err}", pid=paper_id, err=str(e)
                )
                return None
            except Exception as e:
                logger.warning(
                    "Download failed for {pid}: {err}", pid=paper_id, err=str(e)
                )
                return None
            dt_download = time.monotonic() - t0

            # extract and validate
            t1 = time.monotonic()
            try:
                text = await self._extract_and_validate(pdf_path, paper_id)
            except PdfExtractionError as e:
                logger.warning(
                    "Extraction failed for {pid}: {err}", pid=paper_id, err=str(e)
                )
                return None
            except Exception as e:
                logger.warning(
                    "Unexpected extraction error for {pid}: {err}",
                    pid=paper_id,
                    err=str(e),
                )
                return None
            dt_extract = time.monotonic() - t1

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
        self,
        paper_id: str,
        url: str,
        session: AsyncSession,
        destination_path: str,
    ) -> None:
        """Stream-download a single PDF to the destination path.

        Raises RequestsError for network issues or HTTP status >= 400.
        """
        resp = await session.get(url, stream=True)
        status = resp.status_code
        if status >= 400:
            # log specific HTTP errors
            logger.warning(
                "HTTP {status} when fetching PDF for {pid}",
                status=status,
                pid=paper_id,
            )
            raise RequestsError(f"HTTP {status}")

        with open(destination_path, "wb") as fh:
            async for chunk in resp.aiter_content(chunk_size=self._CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)

    async def _extract_and_validate(self, tmp_path: str, paper_id: str) -> str:
        """Run PyMuPDF extraction in a thread and validate text length.

        Raises PdfExtractionError on invalid/short extraction.
        """
        text = await asyncio.to_thread(extract_text_from_pdf_file, tmp_path)
        if not text or len(text) < 500:
            raise PdfExtractionError("Extracted text too short (<500 chars)")
        return text


__all__ = ["AcquisitionAgent"]
