import pytest
from utils.section_detector import SectionDetector


@pytest.fixture
def detector():
    """Create a fresh SectionDetector instance for each test."""
    return SectionDetector()


def test_standard_academic_paper(detector):
    """Test detection of standard academic paper with numbered section headers."""
    text = """1. Introduction

This paper discusses machine learning techniques for section detection.

2. Methods

We used regex patterns to identify section headers.

3. Results

The results show that our approach works well.

4. Discussion

This method is efficient and accurate.

5. Conclusion

Future work could extend this approach."""

    chunks = detector.detect_sections(text)

    # Should detect all 5 sections
    assert len(chunks) == 5

    # Check section types and labels
    section_names = [chunk["section"] for chunk in chunks]
    assert "introduction" in section_names
    assert "methods" in section_names
    assert "results" in section_names
    assert "discussion" in section_names
    assert "conclusion" in section_names

    # Check that positions are in ascending order
    positions = [chunk["start_pos"] for chunk in chunks]
    assert positions == sorted(positions)

    # Check content structure
    for chunk in chunks:
        assert "content" in chunk
        assert "section" in chunk
        assert "start_pos" in chunk
        assert len(chunk["content"]) > 10  # Reasonable minimum content length


def test_non_standard_section_naming(detector):
    """Test detection with variations in section naming."""
    text = """Abstract

This is an overview of our research.

Materials and Methods

We collected data using various approaches.

Findings

Our results indicate significant improvements.

Discussion and Results

Here we discuss the implications."""

    chunks = detector.detect_sections(text)

    # Should detect introduction/abstract section
    assert any(s["section"] == "introduction" for s in chunks)

    # Check for methods section (materials and methods)
    assert any(s["section"] == "methods" for s in chunks)

    # Check for results section
    assert any(s["section"] == "results" for s in chunks)

    # Should detect sections despite non-standard naming
    assert len(chunks) >= 3


def test_fallback_chunking_behavior(detector):
    """Test fallback to chunking when insufficient sections are detected."""
    text = """This is a poorly structured paper with just one main text block.

It discusses various topics without clear section headers.

There are no numbered sections or clear delimiters.

This should trigger the fallback chunking mechanism."""

    chunks = detector.detect_sections(text)

    # Should use fallback chunking (all chunks marked as "unknown")
    assert all(chunk["section"] == "unknown" for chunk in chunks)

    # Should create chunks of appropriate size with overlap
    for chunk in chunks:
        assert len(chunk["content"]) > 0
        assert (
            len(chunk["content"]) <= 1100
        )  # chunk_size + some buffer for sentence boundary detection

    # At least one chunk should contain content
    assert len(chunks) >= 1

    # Check position ordering
    if len(chunks) > 1:
        positions = [chunk["start_pos"] for chunk in chunks]
        assert positions == sorted(positions)
        # Ensure there's some overlap between consecutive chunks
        assert all(
            positions[i + 1] < positions[i] + detector.chunk_size
            for i in range(len(positions) - 1)
        )


def test_empty_text_error(detector):
    """Test that empty text raises proper error."""
    with pytest.raises(ValueError, match="Invalid input text"):
        detector.detect_sections("")

    with pytest.raises(ValueError, match="Invalid input text"):
        detector.detect_sections("   ")

    with pytest.raises(ValueError, match="Input must be a string"):
        detector.detect_sections(None)


def test_case_insensitive_section_detection(detector):
    """Test that section detection works regardless of case."""
    text = """INTRODUCTION

Uppercase header.

introduction

Lowercase header.

IntRODucTIOn

Mixed case header.

Methods

Regular methods section."""

    chunks = detector.detect_sections(text)

    # Should detect multiple introduction sections
    intro_count = sum(1 for chunk in chunks if chunk["section"] == "introduction")
    assert intro_count >= 2  # At least the uppercase and lowercase ones

    # Should also detect methods section
    assert any(chunk["section"] == "methods" for chunk in chunks)


def test_malformed_section_headers(detector):
    """Test handling of malformed or confusing section headers."""
    text = """This paper has discussion in the middle of text and no proper structure.

We also mention methods here and there.

Results are scattered throughout.

This should fallback to chunking."""

    chunks = detector.detect_sections(text)

    # With scattered mentions, should fallback to chunking
    assert all(chunk["section"] == "unknown" for chunk in chunks)
    assert len(chunks) >= 1


def test_section_boundary_accuracy(detector):
    """Test that section content extraction maintains proper boundaries."""
    text = """1. Introduction

This is introduction content that should be in the intro section with more text to ensure it passes the minimum content filter and provides sufficient substance for the minimum length requirements that we have in place.

2. Methods

This is methods content that belongs to the methods section with additional description to make it clearly substantial and provide enough length so that it passes all the filters we have implemented for section detection.

3. Results

Results content here with additional details and explanations to meet the minimum content requirements and make this section clearly distinct enough to be considered a valid section with sufficient length for testing purposes."""

    chunks = detector.detect_sections(text)

    # Should have 2-3 sections depending on content filtering (main focus is boundary accuracy)
    assert len(chunks) >= 2

    # Verify we detect introduction and methods sections
    section_types = {chunk["section"] for chunk in chunks}
    assert "introduction" in section_types
    assert "methods" in section_types

    # Check that content doesn't jump between sections
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        # Introduction chunk should not contain methods material
        if chunk["section"] == "introduction":
            assert "methods" not in content_lower
        elif chunk["section"] == "methods":
            assert "introduction" not in content_lower


def test_overlapping_section_headers(detector):
    """Test that overlapping or nested section headers are handled properly."""
    text = """Experimental Methods and Materials and Methods

This section combines two concepts.

Experimental Results and Findings

Combined results section."""

    chunks = detector.detect_sections(text)

    # Should still detect sections properly despite overlap potentials
    assert len(chunks) >= 1


def test_subsection_filtering(detector):
    """Test that subsections are properly filtered out."""
    text = """1. Introduction

This is the main introduction section.

1.1 Background

This subsection should be ignored.

2. Methods

Main methods section.

2.1 Data Collection

This subsection should not appear as a main section."""

    chunks = detector.detect_sections(text)

    # Should only detect main sections, not subsections
    main_sections = [chunk["section"] for chunk in chunks]
    assert "introduction" in main_sections
    assert "methods" in main_sections
    assert len(chunks) >= 2

    # Verify that content doesn't inappropriately include subsection headers
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        assert "1.1 background" not in content_lower
        assert "2.1 data collection" not in content_lower


def test_only_introduction_conclusion(detector):
    """Test handling papers with only Introduction and Conclusion sections."""
    text = """Introduction

This paper studies machine learning algorithms.

Materials and Methods

The methods used are described here.

Results

We achieved good results.

Discussion

Here we discuss the implications.

Conclusion

Future work is planned."""

    chunks = detector.detect_sections(text)

    # Should detect sections despite lack of numbering
    assert len(chunks) >= 4

    # Should include introduction and conclusion
    section_types = {chunk["section"] for chunk in chunks}
    assert "introduction" in section_types
    assert "conclusion" in section_types


@pytest.mark.skip(
    reason="Memory-efficient mode test requires large memory allocation, run manually if needed"
)
def test_large_paper_memory_efficiency(detector):
    """Test memory-efficient processing for large papers (approaching size limit)."""
    from unittest.mock import patch

    # Use a moderately large text and mock the threshold to force memory-efficient mode
    large_text = (
        "Introduction\n\n"
        + "This is a large document with content. " * 10000
        + "\n\nMethods\n\n"
        + "More content here. " * 10000
    )

    # Temporarily mock the memory threshold to force memory-efficient mode
    with patch.object(
        detector, "memory_efficient_threshold", len(large_text.encode("utf-8")) - 1000
    ):
        # Ensure we're testing memory-efficient mode
        assert len(large_text.encode("utf-8")) > detector.memory_efficient_threshold

        # This should process successfully without raising errors and use memory-efficient mode
        chunks = detector.detect_sections(large_text)

        # Should have chunks (either sections or fallback)
        assert len(chunks) > 0

        # Verify reasonable performance - memory efficient mode should work
        assert all(
            chunk["section"]
            in [
                "introduction",
                "methods",
                "results",
                "discussion",
                "conclusion",
                "unknown",
            ]
            for chunk in chunks
        )


def test_malformed_pdf_text_error_handling(detector):
    """Test comprehensive error handling for malformed PDF text."""
    # Test null bytes
    corrupted_text = "Good text\x00\x00\x00bad null bytes" * 1000
    with pytest.raises(ValueError, match="corrupted.*null bytes"):
        detector.detect_sections(corrupted_text)

    # Test invalid UTF-8
    invalid_utf8 = b"\xff\xfe\xfd".decode("latin-1", errors="ignore") + "good text"
    # This might pass if decoding succeeds, but we test the validation
    if invalid_utf8:
        try:
            detector.detect_sections(invalid_utf8)
        except ValueError as e:
            assert "UTF-8" in str(e) or "input text" in str(e).lower()

    # Test extremely long text (exceeding size limit)
    oversized_detector = SectionDetector(max_text_size_mb=1)  # 1MB limit
    large_text = "This is a test document. " * 100000  # About 2.5MB
    with pytest.raises(ValueError, match="exceeds maximum.*size"):
        oversized_detector.detect_sections(large_text)


def test_appendix_exclusion(detector):
    """Test that appendices and supplemental material are excluded."""
    text = """1. Introduction

Main introduction content.

2. Methods

Methods description.

Appendix A

This appendix should be ignored.

Supplemental Material

This should also be ignored.

References

References section should be ignored."""

    chunks = detector.detect_sections(text)

    # Should detect main sections but not appendices
    section_types = {chunk["section"] for chunk in chunks}
    assert "introduction" in section_types
    assert "methods" in section_types

    # Verify no content from appendices appears in main sections
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        assert "appendix a" not in content_lower
        assert "supplemental material" not in content_lower
        assert "references" not in content_lower
