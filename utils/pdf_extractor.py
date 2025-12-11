from __future__ import annotations

from typing import Optional
import fitz  # PyMuPDF


class PDFExtractionError(Exception):
    """Base class for PDF extraction errors."""

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


# Backwards compatibility alias
PdfExtractionError = PDFExtractionError


def extract_text_from_pdf_file(path: str, min_length: int = 500) -> str:
    """Extract text from a PDF file path using standard PyMuPDF (Fast).
    Returns plain text joined by newlines.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise PdfExtractionError(f"Failed to open PDF file: {e}") from e

    try:
        page_count = doc.page_count
        if page_count == 0:
            raise EmptyPDFError("Empty PDF: 0 pages found")

        texts = []
        for i in range(page_count):
            # "text" mode is instant (no layout analysis)
            page = doc.load_page(i)
            texts.append(page.get_text("text").strip())

        text = "\n\n".join(texts).strip()

    except Exception as e:
        raise PdfExtractionError(f"PDF extraction failed: {e}") from e
    finally:
        try:
            doc.close()
        except Exception:
            pass

    # Validation
    if not text or len(text) < min_length:
        raise InsufficientTextError(f"Extracted text too short (<{min_length} chars)")

    return text


__all__ = [
    "extract_text_from_pdf_file",
    "PdfExtractionError",
    "CorruptedPDFError",
    "EmptyPDFError",
    "InsufficientTextError",
]
