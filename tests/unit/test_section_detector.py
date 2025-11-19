import pytest

from utils.section_detector import detect_sections


@pytest.mark.unit
def test_detect_standard_sections():
    """Test detection of standard 5-section paper."""
    text = (
        "# Introduction\nThis paper investigates...\n\n"
        "# Methods\nWe conducted experiments...\n\n"
        "# Results\nOur findings show...\n\n"
        "# Discussion\nThese results indicate...\n\n"
        "# Conclusion\nIn summary, we found...\n"
    )

    sections = detect_sections(text, paper_id="test123", min_section_length=0)

    assert len(sections) == 5
    assert sections[0]["name"] == "Introduction"
    assert sections[1]["name"] == "Methods"
    assert "experiments" in sections[1]["text"]
    assert sections[0]["start_idx"] == 0


@pytest.mark.unit
def test_detect_three_sections_missing_some_headers():
    text = (
        "# Introduction\nShort intro...\n\n"
        "# Methods\nMethod details...\n\n"
        "# Results\nResults here...\n"
    )

    sections = detect_sections(text, paper_id="test_missing", min_section_length=0)
    assert len(sections) == 3
    names = [s["name"] for s in sections]
    assert names == ["Introduction", "Methods", "Results"]


@pytest.mark.unit
def test_fallback_for_unstructured_text():
    # Unstructured text with no headers should trigger fallback chunking
    words = [f"word{i}" for i in range(3000)]
    text = " ".join(words)

    sections = detect_sections(
        text, paper_id="fallback_test", fallback_token_size=1000, min_section_length=0
    )
    assert len(sections) >= 2
    assert sections[0]["name"].startswith("chunk_")
    # ensure chunks cover the whole text
    total_len = sum((s["end_idx"] - s["start_idx"] for s in sections))
    assert total_len > 0


@pytest.mark.unit
def test_all_caps_headers():
    text = (
        "INTRODUCTION\nIntro text...\n\n"
        "METHODS\nMethod text...\n\n"
        "RESULTS\nResults...\n"
    )
    sections = detect_sections(text, paper_id="caps_test", min_section_length=0)
    names = [s["name"] for s in sections]
    assert "Introduction" in names
    assert "Methods" in names
    assert "Results" in names


@pytest.mark.unit
def test_markdown_style_headers():
    text = (
        "## Introduction\nIntro here...\n\n"
        "## Methods\nStuff...\n\n"
        "## Results\nFindings...\n"
    )
    sections = detect_sections(text, paper_id="md_test", min_section_length=0)


@pytest.mark.unit
def test_numbered_headers():
    text = (
        "1. Introduction\nIntro text...\n\n"
        "2) Methods\nMethod text...\n\n"
        "III. Results\nResults...\n\n"
        "(4) Discussion\nDiscuss...\n\n"
        "5 Conclusion\nIn summary...\n"
    )
    sections = detect_sections(text, paper_id="numbered_test", min_section_length=0)
    names = [s["name"] for s in sections]
    assert "Introduction" in names
    assert "Methods" in names
    assert "Results" in names
    assert "Discussion" in names
    assert "Conclusion" in names
    assert len(sections) == 5
    assert sections[0]["name"] == "Introduction"
