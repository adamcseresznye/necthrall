from __future__ import annotations

from typing import Optional, Tuple

import time
from loguru import logger
import fitz  # PyMuPDF
import pymupdf4llm


class PDFExtractionError(Exception):
    """Base class for PDF extraction errors.

    All custom PDF extraction exceptions inherit from this class. The
    `paper_id` is included in the message when provided to aid debugging
    and structured logs.
    """

    def __init__(self, message: str, paper_id: Optional[str] = None):
        if paper_id:
            message = f"[paper_id={paper_id}] {message}"
        super().__init__(message)


class CorruptedPDFError(PDFExtractionError):
    """Raised when PDF bytes are corrupted, unreadable, or encrypted."""


class EmptyPDFError(PDFExtractionError):
    """Raised when the PDF contains zero pages."""


class InsufficientTextError(PDFExtractionError):
    """Raised when extracted text is below the configured minimum length."""


# Backwards compatibility alias used by older callers in the codebase
PdfExtractionError = PDFExtractionError


def _extract_pages(doc: fitz.Document) -> Tuple[str, int]:
    """Extract text as Markdown for all pages in a PyMuPDF document.

    Uses PyMuPDF4LLM to extract content as GitHub-compatible Markdown,
    preserving document structure including:
    - Headers (detected via font size)
    - Tables (in machine-readable format)
    - Multi-column reading order
    - Section hierarchy

    Returns:
        A tuple of (markdown_text, page_count).
    """
    page_count = doc.page_count

    try:
        # PyMuPDF4LLM's to_markdown works on the doc object directly
        markdown_text = pymupdf4llm.to_markdown(doc)
        if markdown_text and markdown_text.strip():
            return markdown_text.strip(), page_count
    except Exception as e:
        # If PyMuPDF4LLM fails, fall back to plain text extraction
        logger.warning(
            f"PyMuPDF4LLM extraction failed, falling back to plain text: {e}"
        )
        pass

    # Fallback: per-page plain text extraction
    texts = []
    for i in range(page_count):
        page = doc.load_page(i)
        texts.append(page.get_text("text").rstrip())

    return "\n".join(texts).strip(), page_count


def _validate_text(text: str, min_length: int, paper_id: Optional[str] = None) -> None:
    """Validate extracted text length and raise if insufficient.

    Raises:
        InsufficientTextError: if the text length is below `min_length`.
    """
    length = len(text or "")
    if length < min_length:
        raise InsufficientTextError(
            f"Insufficient text: {length} chars (min {min_length})", paper_id=paper_id
        )


def extract_text_from_pdf(
    pdf_bytes: bytes, paper_id: Optional[str] = None, min_length: int = 500
) -> str:
    """Extract text from PDF bytes using PyMuPDF.

    This function is synchronous and side-effect free: it opens the PDF
    from bytes, extracts text, validates length, and returns the text.

    Example:
        text = extract_text_from_pdf(pdf_bytes, paper_id="doi:...", min_length=500)

    Args:
        pdf_bytes: Raw PDF file content
        paper_id: Optional paper ID for logging context
        min_length: Minimum required text length (default 500)

    Returns:
        Extracted text as string (length >= min_length)

    Raises:
        CorruptedPDFError: PDF bytes are invalid, corrupted or encrypted
        EmptyPDFError: PDF contains zero pages
        InsufficientTextError: Extracted text length < min_length
    """
    ctx = {"paper_id": paper_id} if paper_id else {}
    logger.debug(
        "Starting PDF extraction",
        **{**ctx, "bytes": len(pdf_bytes) if pdf_bytes is not None else 0},
    )

    start = time.perf_counter()
    try:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            logger.exception("PDF open failed: corrupted PDF bytes", **ctx)
            raise CorruptedPDFError(
                "Corrupted PDF: invalid or unreadable bytes", paper_id=paper_id
            ) from e
        except Exception as e:
            msg = str(e).lower()
            if "encrypted" in msg or "password" in msg:
                logger.exception(
                    "PDF open failed: password-protected or encrypted", **ctx
                )
                raise CorruptedPDFError(
                    "Password-protected or encrypted PDF", paper_id=paper_id
                ) from e
            logger.exception("PDF open failed: unexpected error", **ctx)
            raise CorruptedPDFError(
                f"Failed to open PDF: {e}", paper_id=paper_id
            ) from e

        # get page count and guard
        try:
            page_count = doc.page_count
        except Exception as e:
            logger.exception("Failed to read page count", **ctx)
            raise CorruptedPDFError(
                "Unable to determine PDF page count", paper_id=paper_id
            ) from e

        if page_count == 0:
            logger.exception("Empty PDF: 0 pages found", **ctx)
            raise EmptyPDFError("Empty PDF: 0 pages found", paper_id=paper_id)

        # If the document reports as encrypted/protected, surface as corrupted
        try:
            needs_pass = getattr(doc, "needs_pass", False) or getattr(
                doc, "is_encrypted", False
            )
            if needs_pass:
                logger.exception("Password-protected PDF detected", **ctx)
                raise CorruptedPDFError(
                    "Password-protected or encrypted PDF", paper_id=paper_id
                )
        except Exception:
            # Non-fatal if attribute access fails; continue to attempt extraction
            pass

        # Extract pages (fast path tries full-document extraction)
        full_text, extracted_pages = _extract_pages(doc)
    except PDFExtractionError:
        raise
    except Exception as e:
        logger.exception("Failed during PDF processing", **ctx)
        raise CorruptedPDFError(
            f"Error extracting text from PDF: {e}", paper_id=paper_id
        ) from e
    finally:
        try:
            # ensure doc is closed if it was opened
            if "doc" in locals():
                doc.close()
        except Exception:
            pass

    # Validate extracted text
    try:
        _validate_text(full_text, min_length, paper_id=paper_id)
    except InsufficientTextError:
        logger.exception(
            "Insufficient text extracted",
            **{**ctx, "chars": len(full_text or ""), "pages": extracted_pages},
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.debug(
        "PDF text extracted successfully",
        **{
            **ctx,
            "chars": len(full_text or ""),
            "pages": extracted_pages,
            "ms": round(elapsed_ms, 2),
        },
    )

    return full_text


def extract_text_from_pdf_bytes(pdf_bytes: bytes, min_length: int = 500) -> str:
    """Backward-compatible wrapper returning text or raising PdfExtractionError."""
    try:
        return extract_text_from_pdf(pdf_bytes, paper_id=None, min_length=min_length)
    except (CorruptedPDFError, EmptyPDFError, InsufficientTextError) as e:
        raise PdfExtractionError(str(e)) from e


def extract_text_from_pdf_file(path: str, min_length: int = 500) -> str:
    """Extract text from a PDF file path.

    Raises PdfExtractionError on failure so callers written for the original
    API behave the same.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise PdfExtractionError(f"Failed to open PDF file: {e}") from e

    try:
        page_count = doc.page_count
        if page_count == 0:
            raise PdfExtractionError("Empty PDF: 0 pages found")

        texts = []
        for i in range(page_count):
            page = doc.load_page(i)
            texts.append(page.get_text("text").rstrip())
        text = "\n".join(texts).strip()
    except Exception as e:
        raise PdfExtractionError(f"PDF extraction failed: {e}") from e
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if not text or len(text) < min_length:
        raise PdfExtractionError("Extracted text too short (<500 chars)")

    return text


__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_pdf_bytes",
    "extract_text_from_pdf_file",
    "PdfExtractionError",
    "CorruptedPDFError",
    "EmptyPDFError",
    "InsufficientTextError",
]
