import pytest
from agents.acquisition import (
    PDFTextExtractor,
    PDFParsingError,
    InvalidContentError,
)
import fitz
from models.state import PDFContent


@pytest.fixture
def extractor():
    """Provides a PDFTextExtractor instance for testing."""
    return PDFTextExtractor()


def test_successful_extraction(extractor):
    """Tests successful text extraction from a valid PDF."""
    long_text = "This is a test sentence designed to be well over one hundred characters to validate the successful extraction of text content from a standard, well-formed PDF document."
    pdf_bytes = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 52 >>\nstream\nBT\n/F1 12 Tf\n100 100 Td\n("
        + long_text.encode()
        + b") Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n0000000112 00000 n \n0000000195 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n258\n%%EOF"
    )
    result = extractor.extract("test_pdf_success", pdf_bytes)
    assert isinstance(result, PDFContent)
    assert result.paper_id == "test_pdf_success"
    assert result.char_count > 100
    assert "well over one hundred characters" in result.raw_text


def test_corrupted_pdf_handling(extractor):
    """Tests graceful failure with corrupted PDF bytes."""
    corrupted_bytes = b"%PDF-1.4\n%%corrupted\n"
    with pytest.raises(PDFParsingError):
        extractor.extract("corrupted_pdf", corrupted_bytes)


def test_password_protected_pdf_skipping(extractor):
    """Tests that password-protected PDFs are skipped gracefully."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 52 >>\nstream\nBT\n/F1 12 Tf\n100 100 Td\n(This is a protected PDF.) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n0000000112 00000 n \n0000000195 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R /Encrypt << /V 1 /R 2 /O (xxx) /U (xxx) /P -4 >> >>\nstartxref\n258\n%%EOF"
    with pytest.raises(PDFParsingError):
        extractor.extract("protected_pdf", pdf_bytes)


def test_zero_page_pdf_handling(extractor):
    """Tests that zero-page PDFs are handled correctly."""
    # Create a valid PDF with zero pages
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n102\n%%EOF"
    with pytest.raises(InvalidContentError):
        extractor.extract("zero_page_pdf", pdf_bytes)


def test_insufficient_text_filtering(extractor):
    """Tests that PDFs with insufficient text are filtered out."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 52 >>\nstream\nBT\n/F1 12 Tf\n100 100 Td\n(Too short.) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n0000000112 00000 n \n0000000195 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n258\n%%EOF"
    with pytest.raises(InvalidContentError):
        extractor.extract("short_pdf", pdf_bytes)


def test_image_only_pdf_detection():
    """Tests the early termination mechanism for image-only PDFs."""
    # This test is a proxy; in a real scenario, we'd use a PDF with actual images.
    # Here, we simulate it with very low text content over several pages.
    doc = fitz.open()
    for _ in range(5):
        page = doc.new_page()
        page.insert_text((50, 72), "a")
    pdf_bytes = doc.write()
    doc.close()
    extractor_low_threshold = PDFTextExtractor(min_text_length=100)
    with pytest.raises(InvalidContentError, match="Insufficient text content"):
        extractor_low_threshold.extract("image_only_pdf", pdf_bytes)


# def test_scientific_character_preservation(extractor):
#     """Tests that scientific and mathematical characters are preserved."""
#     with open("tests/symbols.pdf", "rb") as f:
#         pdf_bytes = f.read()
#     extractor_low_threshold = PDFTextExtractor(min_text_length=10)
#     result = extractor_low_threshold.extract("scientific_pdf", pdf_bytes)
#     assert isinstance(result, PDFContent)
#     assert "α" in result.raw_text
#     assert "β" in result.raw_text
#     assert "γ" in result.raw_text
#     assert "±" in result.raw_text
#     assert "∑" in result.raw_text
#     assert "∫" in result.raw_text
#     assert "∞" in result.raw_text
#     assert "≠" in result.raw_text
#     assert "≤" in result.raw_text
#     assert "≥" in result.raw_text


def test_text_cleaning_logic(extractor):
    """Tests the internal text cleaning logic."""
    raw_text = (
        "  This   is a \t test with \n extra whitespace and \x0c control chars.  "
    )
    cleaned_text = extractor._clean_text(raw_text)
    assert cleaned_text == "This is a test with extra whitespace and control chars."
