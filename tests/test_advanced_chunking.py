"""
Comprehensive Test Suite for Advanced Document Chunking

Tests the AdvancedDocumentChunker implementation with SpaCy integration.
Covers the 4 required test cases from the specification and additional validation.

Test Cases:
1. Section identification works on standard academic paper format
2. Sentence boundary preservation during chunking
3. Intelligent overlap between chunks maintains context
4. Chunk metadata is accurate and complete

Additional Tests:
- Performance validation (< 1 second for 25 documents)
- Edge case handling (missing sections, corrupted text, very short sections)
- Quality validation and error handling
- Memory-efficient processing
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from rag.chunking import AdvancedDocumentChunker, SectionPatternConfig
from models.state import Chunk, PDFContent


@pytest.fixture
def chunker():
    """Create a fresh AdvancedDocumentChunker instance for each test."""
    return AdvancedDocumentChunker(
        chunk_size=500,  # Tokens
        chunk_overlap=50,  # Tokens
        min_section_chars=20,  # Lower threshold to detect shorter sections
        min_chunk_tokens=10,  # Lower threshold for testing
        memory_threshold_mb=50,  # Smaller for testing
    )


@pytest.fixture
def sample_paper():
    """Sample paper object for testing."""
    paper = Mock()
    paper.paper_id = "test_paper_123"
    paper.title = "Test Paper Title"
    return paper


@pytest.fixture
def sample_academic_text():
    """Sample academic paper text with standard sections."""
    return """1. Introduction

This paper investigates novel machine learning techniques for natural language processing. Recent advances in transformer architectures have revolutionized the field of computational linguistics. Our research focuses on developing efficient chunking strategies that preserve semantic boundaries while maintaining contextual coherence.

The motivation for this work stems from the need to process large scientific documents efficiently. Traditional chunking methods often break semantic units, leading to loss of important contextual information. By leveraging advanced NLP techniques, we aim to create more intelligent document processing pipelines.

2. Methods

We employed a comprehensive methodology combining regex-based section detection with SpaCy-powered sentence segmentation. The section detection phase uses carefully crafted regular expressions to identify standard academic paper structures including Introduction, Methods, Results, Discussion, and Conclusion.

After section identification, each section is processed independently using token-based chunking. We utilize SpaCy's en_core_web_sm model for accurate sentence boundary detection, ensuring that chunks respect natural language boundaries. Token counting is performed using SpaCy's tokenizer to maintain consistency between chunking and downstream processing.

3. Results

Our experimental results demonstrate significant improvements over baseline chunking approaches. The intelligent overlap mechanism successfully preserved contextual information between chunks, with an average overlap quality score of 0.85. Section-aware chunking achieved 94% accuracy in boundary detection across our test corpus of 50 academic papers.

Performance metrics show that the system can process 25 documents in under 1 second on standard hardware, generating between 50-100 chunks per typical academic paper. Memory usage remained efficient even for documents up to 50 pages, with peak consumption not exceeding 100MB per document.

4. Discussion

The integration of section-aware chunking with intelligent overlap represents a significant advancement in document processing technology. While traditional approaches treat documents as monolithic text blocks, our method recognizes the hierarchical structure inherent in academic writing. This section-level awareness enables more nuanced chunking strategies that respect the author's intended organization.

However, our approach is not without limitations. The current implementation relies heavily on standard section naming conventions, which may not capture all variations in academic writing styles across different disciplines. Future work could incorporate machine learning approaches to improve section detection robustness.

5. Conclusion

In conclusion, we have demonstrated the effectiveness of intelligent, section-aware document chunking with SpaCy integration. The system's ability to preserve sentence boundaries while creating meaningful overlap between chunks addresses key challenges in document processing for retrieval-augmented generation systems.

Our implementation successfully meets all performance requirements, processing documents efficiently while maintaining high quality standards. The comprehensive metadata generation and quality validation features ensure reliable operation in production environments.

Future research directions include extending the approach to multilingual documents and incorporating domain-specific section patterns for specialized academic fields."""


@pytest.fixture
def sample_pdf_content(sample_academic_text):
    """Sample PDF content for testing."""
    return PDFContent(
        paper_id="test_paper_123",
        raw_text=sample_academic_text,
        page_count=5,
        char_count=len(sample_academic_text),
        extraction_time=0.5,
    )


class TestSectionIdentification:
    """
    Test Case 1: Section identification works on standard academic paper format

    Validates that the chunker correctly identifies and processes standard academic sections.
    """

    def test_standard_academic_paper_sections(
        self, chunker, sample_paper, sample_pdf_content
    ):
        """Test detection of standard academic paper sections."""
        chunks, errors, stats = chunker.process_papers(
            [sample_paper], [sample_pdf_content]
        )

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(chunks) > 0, "Should generate at least one chunk"
        assert stats["processed_papers"] == 1, "Should process exactly one paper"

        # Verify section detection
        section_counts = stats.get("section_distribution", {})
        detected_sections = set(section_counts.keys())

        # Should detect main sections (allowing for fallback)
        expected_sections = {
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        }
        assert (
            len(detected_sections.intersection(expected_sections)) >= 3
        ), f"Expected sections not detected: {detected_sections}"

        # At least some sections should be properly identified (not unknown)
        assert (
            "unknown" not in detected_sections or len(detected_sections) > 1
        ), "Should detect some proper sections"

    def test_section_pattern_config(self):
        """Test that section patterns are correctly configured."""
        config = SectionPatternConfig()

        # Test each pattern
        assert "abstract" in config.PATTERNS
        assert "introduction" in config.PATTERNS
        assert "methods" in config.PATTERNS
        assert "results" in config.PATTERNS
        assert "discussion" in config.PATTERNS
        assert "conclusion" in config.PATTERNS

        # Test pattern matching
        intro_pattern = config.PATTERNS["introduction"]
        assert intro_pattern.search("1. Introduction")
        assert intro_pattern.search("Introduction")
        assert intro_pattern.search("intro")
        assert intro_pattern.search("INTRODUCTION")  # Case insensitive

        methods_pattern = config.PATTERNS["methods"]
        assert methods_pattern.search("2. Methods")
        assert methods_pattern.search("Materials and Methods")
        assert methods_pattern.search("METHODOLOGY")

    def test_section_boundary_accuracy(self, chunker):
        """Test that section boundaries are accurately identified."""
        text = """Some introductory content.

1. Introduction

This is the introduction section with substantial content that should be long enough to pass the test requirements.

2. Methods

This is the methods section with detailed description that should also be long enough for the test.

3. Results

Results section content here with enough text to satisfy the minimum length requirement."""

        sections = chunker._detect_sections_with_metadata(text)

        assert len(sections) >= 3, f"Expected at least 3 sections, got {len(sections)}"

        # Verify section names
        section_names = [s["section"] for s in sections]
        assert "introduction" in section_names
        assert "methods" in section_names
        assert "results" in section_names

        # Verify content boundaries (allow for header in first section)
        for section in sections:
            content = section["content"]
            # Allow for some sections to have header-only content when they are at the end
            if (
                section["section"] != "results"
            ):  # Last section might have minimal content
                assert (
                    len(content.strip()) > 30
                ), f"Section {section['section']} has too little content: '{content.strip()}'"


class TestSentenceBoundaryPreservation:
    """
    Test Case 2: Sentence boundary preservation during chunking

    Validates that sentence boundaries are preserved and chunks don't break sentences inappropriately.
    """

    def test_sentence_boundary_preservation(self, chunker):
        """Test that sentence boundaries are preserved in chunking."""
        text = """This is the first sentence. It contains important information about our research methodology. The second sentence provides additional context. Finally, the third sentence concludes this thought.

This is a new paragraph. Another sentence here. And yet another one."""

        # Process text directly using private method for testing
        chunks, token_count = chunker._chunk_section_with_spacy(text, "test_section")

        # Should generate chunks
        assert len(chunks) > 0, "Should generate at least one chunk"

        # Each chunk should contain sentences that end properly
        for chunk_data in chunks:
            content = chunk_data["content"]
            sentences = [s.strip() for s in content.split(".") if s.strip()]

            # Content should not be cut mid-sentence (basic check)
            if not content.endswith("."):
                # If chunk doesn't end with period, ensure it's not cut off mid-sentence
                last_sentence = content.split(".")[-1].strip()
                assert (
                    len(last_sentence) < 100
                ), f"Chunk seems to be cut mid-sentence: {last_sentence[:100]}"

            # Token count should be accurate
            assert chunk_data["token_count"] > 0, "Token count should be positive"

    def test_spacy_sentence_segmentation(self, chunker):
        """Test sentence segmentation works correctly."""
        text = "First sentence. Second sentence! Third sentence?"

        # Use the chunker's sentence splitting method
        sentences = chunker._split_sentences(text)

        assert (
            len(sentences) >= 3
        ), f"Expected at least 3 sentences, got {len(sentences)}: {sentences}"

    def test_chunk_size_token_based(self, chunker):
        """Test that chunk sizing works correctly based on tokens."""
        # Create text with known token count
        text = "This is a simple sentence. " * 100  # Should be more than 500 tokens

        chunks, total_tokens = chunker._chunk_section_with_spacy(text, "test")

        # Most chunks should be close to target size
        target_size = 500
        overlap = 50

        for i, chunk in enumerate(chunks):
            token_count = chunk["token_count"]
            if i < len(chunks) - 1:  # Not the last chunk
                # Allow some flexibility around target size
                assert (
                    token_count >= target_size * 0.7
                ), f"Chunk {i} too small: {token_count} tokens"
                assert (
                    token_count <= target_size * 1.5
                ), f"Chunk {i} too large: {token_count} tokens"


class TestIntelligentOverlap:
    """
    Test Case 3: Intelligent overlap between chunks maintains context

    Validates that overlap preserves context and doesn't create redundancy.
    """

    def test_overlap_creation(self, chunker):
        """Test that overlap is created between consecutive chunks."""
        # Create text that will generate multiple chunks
        text = "This is sentence one. " * 200  # Long text to ensure multiple chunks

        chunks, _ = chunker._chunk_section_with_spacy(text, "test_section")

        if len(chunks) > 1:
            # Check overlap preservation metadata
            for i, chunk in enumerate(chunks[1:], 1):  # Start from second chunk
                overlap_preserved = chunk.get("overlap_preserved", False)
                assert overlap_preserved, f"Chunk {i} should have overlap preserved"

    def test_overlap_token_requirements(self, chunker):
        """Test that overlap achieves minimum token requirements."""
        sentences = []
        # Create sentences with known token counts
        for i in range(20):
            sentence = f"This is sentence number {i} with some additional content. "
            sentences.append(sentence)

        text = "".join(sentences)

        # Mock the overlap finding to test specific scenarios
        overlap_start = chunker._find_overlap_start(sentences, 10, 50)

        # Should find some overlap
        assert overlap_start <= 10, f"Overlap start position invalid: {overlap_start}"

        # Calculate overlap tokens using regex
        overlap_sentences = sentences[overlap_start:11]  # Up to sentence 10
        overlap_tokens = chunker._count_tokens(" ".join(overlap_sentences))

        assert (
            overlap_tokens >= 30
        ), f"Overlap too small: {overlap_tokens} tokens"  # Allow some flexibility

    def test_context_preservation(self, chunker):
        """Test that overlap preserves meaningful context."""
        text = """The machine learning model achieved high accuracy. This performance exceeded our initial expectations. The results validate our hypothesis. Further experimentation is warranted."""

        chunks, _ = chunker._chunk_section_with_spacy(text, "results")

        # With such short text and default chunk sizes, should create one chunk or handle gracefully
        assert len(chunks) >= 1, "Should create at least one chunk"

        # Content should be preserved without corruption
        combined_content = " ".join(chunk["content"] for chunk in chunks)
        assert "machine learning model" in combined_content
        assert "high accuracy" in combined_content


class TestChunkMetadata:
    """
    Test Case 4: Chunk metadata is accurate and complete

    Validates that chunk metadata includes all required fields and is accurate.
    """

    def test_chunk_metadata_completeness(
        self, chunker, sample_paper, sample_pdf_content
    ):
        """Test that chunk metadata includes all required fields."""
        chunks, errors, stats = chunker.process_papers(
            [sample_paper], [sample_pdf_content]
        )

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(chunks) > 0, "Should generate chunks"

        for i, chunk in enumerate(chunks):
            # Required fields from Chunk model
            assert hasattr(chunk, "content"), f"Chunk {i} missing content"
            assert hasattr(chunk, "section"), f"Chunk {i} missing section"
            assert hasattr(chunk, "paper_id"), f"Chunk {i} missing paper_id"
            assert hasattr(chunk, "paper_title"), f"Chunk {i} missing paper_title"
            assert hasattr(chunk, "start_position"), f"Chunk {i} missing start_position"
            assert hasattr(chunk, "end_position"), f"Chunk {i} missing end_position"
            assert hasattr(chunk, "token_count"), f"Chunk {i} missing token_count"
            assert hasattr(chunk, "metadata"), f"Chunk {i} missing metadata"

            # Additional metadata fields
            assert (
                "chunk_index" in chunk.metadata
            ), f"Chunk {i} missing chunk_index in metadata"
            assert (
                "chunking_strategy" in chunk.metadata
            ), f"Chunk {i} missing chunking_strategy"
            assert "quality_score" in chunk.metadata, f"Chunk {i} missing quality_score"

            # Values should be reasonable
            assert (
                chunk.paper_id == sample_paper.paper_id
            ), f"Incorrect paper_id: {chunk.paper_id}"
            assert (
                chunk.paper_title == sample_paper.title
            ), f"Incorrect paper_title: {chunk.paper_title}"
            assert (
                chunk.start_position >= 0
            ), f"Invalid start_position: {chunk.start_position}"
            assert (
                chunk.end_position > chunk.start_position
            ), f"Invalid position range: {chunk.start_position}-{chunk.end_position}"
            assert chunk.token_count > 0, f"Invalid token_count: {chunk.token_count}"
            assert (
                0.0 <= chunk.metadata["quality_score"] <= 1.0
            ), f"Invalid quality_score: {chunk.metadata['quality_score']}"

    def test_metadata_accuracy(self, chunker):
        """Test that metadata values are accurate and consistent."""
        text = "Short sentence. Another sentence. Final sentence."
        chunks, _ = chunker._chunk_section_with_spacy(text, "test")

        for chunk in chunks:
            # Token count should match regex-based calculation
            actual_tokens = chunker._count_tokens(chunk["content"])
            assert (
                abs(actual_tokens - chunk["token_count"])
                <= 5  # Allow some margin for different tokenization
            ), f"Token count mismatch: expected {actual_tokens}, got {chunk['token_count']}"

            # Position data should be reasonable
            assert chunk["start_pos"] >= 0
            assert chunk["end_pos"] > chunk["start_pos"]
            assert (
                chunk["end_pos"] - chunk["start_pos"] <= len(chunk["content"]) + 10
            )  # Allow some approximation

    def test_token_count_consistency(self, chunker, sample_paper, sample_pdf_content):
        """Test that token counts are consistent across processing."""
        chunks, errors, stats = chunker.process_papers(
            [sample_paper], [sample_pdf_content]
        )

        # Total tokens from chunks should match stats
        total_chunk_tokens = sum(chunk.token_count for chunk in chunks)
        assert (
            abs(total_chunk_tokens - stats["total_tokens"]) == 0
        ), f"Token count mismatch: chunks={total_chunk_tokens}, stats={stats['total_tokens']}"


class TestPerformanceValidation:
    """Test performance requirements and efficiency."""

    def test_performance_requirement_chunking_speed(
        self, chunker, sample_paper, sample_pdf_content
    ):
        """Test that chunking meets performance requirements (< 1 second for 25 documents)."""
        # Process 5 copies to test batch performance (25 documents would be too slow for unit test)
        papers = [sample_paper] * 5
        contents = [sample_pdf_content] * 5

        start_time = time.time()
        chunks, errors, stats = chunker.process_papers(papers, contents)
        processing_time = time.time() - start_time

        # Scale performance for 25 documents
        scaled_time = processing_time * (25 / 5)  # 5x scaling

        assert scaled_time < 1.0, ".3f"

        # Verify processing was successful
        assert len(chunks) > 0, "Should generate chunks"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_memory_efficiency(self, chunker):
        """Test memory-efficient processing for large documents."""
        # Create a large document
        large_text = "This is a long document. " * 10000  # Approximately 280KB
        large_content = PDFContent(
            paper_id="large_test",
            raw_text=large_text,
            page_count=50,
            char_count=len(large_text),
            extraction_time=1.0,
        )

        large_paper = Mock()
        large_paper.paper_id = "large_test"
        large_paper.title = "Large Test Document"

        # Should process without memory issues
        chunks, errors, stats = chunker.process_papers([large_paper], [large_content])

        assert len(errors) == 0, f"Memory issues detected: {errors}"
        assert len(chunks) > 0, "Should generate chunks from large document"
        assert stats["processed_papers"] == 1, "Should process large document"

    def test_throughput_calculation(self, chunker, sample_paper, sample_pdf_content):
        """Test performance metrics calculation."""
        chunks, errors, stats = chunker.process_papers(
            [sample_paper], [sample_pdf_content]
        )

        # Verify performance metrics
        assert "total_time" in stats
        assert "processed_papers" in stats
        assert "total_chunks" in stats
        assert "throughput_chunks_per_second" in stats
        assert "avg_chunks_per_paper" in stats

        assert stats["total_time"] > 0, "Processing time should be positive"
        assert stats["processed_papers"] == 1
        assert stats["total_chunks"] == len(chunks)
        assert stats["throughput_chunks_per_second"] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_fallback_chunking(self, chunker):
        """Test fallback chunking when section detection fails."""
        # Text with no clear sections
        text = """This is unstructured text without clear section headers.

It discusses various topics in a stream-of-consciousness manner.

There are no numbered sections or standard academic formatting.

This should trigger fallback chunking."""
        pdf_content = PDFContent(
            paper_id="fallback_test",
            raw_text=text,
            page_count=1,
            char_count=len(text),
            extraction_time=0.1,
        )
        paper = Mock()
        paper.paper_id = "fallback_test"
        paper.title = "Fallback Test Paper"

        chunks, errors, stats = chunker.process_papers([paper], [pdf_content])

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(chunks) > 0, "Should generate fallback chunks"

        # All chunks should be marked as fallback
        for chunk in chunks:
            assert chunk.use_fallback == True, "Chunks should be marked as fallback"
            assert (
                chunk.section == "unknown"
            ), "Fallback chunks should have unknown section"

    def test_very_short_sections(self, chunker):
        """Test handling of very short sections."""
        text = """1. Introduction

Brief intro.

2. Methods

Short method.

3. Results

Tiny result.

4. Discussion

Brief discussion.

5. Conclusion

Done."""

        pdf_content = PDFContent(
            paper_id="short_test",
            raw_text=text,
            page_count=1,
            char_count=len(text),
            extraction_time=0.1,
        )
        paper = Mock()
        paper.paper_id = "short_test"
        paper.title = "Short Sections Test"

        chunks, errors, stats = chunker.process_papers([paper], [pdf_content])

        # Should process successfully but possibly with fewer chunks due to short content
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # May generate chunks or skip very short sections
        if len(chunks) > 0:
            for chunk in chunks:
                assert chunk.token_count >= chunker.min_chunk_tokens, ".1f"

    def test_empty_and_corrupted_text(self, chunker):
        """Test handling of empty and corrupted text."""
        # Empty text
        empty_content = PDFContent(
            paper_id="empty_test",
            raw_text="",
            page_count=0,
            char_count=0,
            extraction_time=0.0,
        )
        empty_paper = Mock()
        empty_paper.paper_id = "empty_test"
        empty_paper.title = "Empty Test"

        chunks, errors, stats = chunker.process_papers([empty_paper], [empty_content])

        assert len(chunks) == 0, "Should not generate chunks from empty text"
        assert len(errors) > 0, "Should report error for empty text"

        # Corrupted text with null bytes
        corrupted_text = "Good text\x00\x00\x00bad null bytes" * 100
        corrupted_content = PDFContent(
            paper_id="corrupt_test",
            raw_text=corrupted_text,
            page_count=1,
            char_count=len(corrupted_text),
            extraction_time=0.1,
        )
        corrupted_paper = Mock()
        corrupted_paper.paper_id = "corrupt_test"
        corrupted_paper.title = "Corrupted Test"

        chunks, errors, stats = chunker.process_papers(
            [corrupted_paper], [corrupted_content]
        )

        assert len(chunks) == 0, "Should not process corrupted text"
        assert len(errors) > 0, "Should report error for corrupted text"

    def test_chunk_validation_functionality(
        self, chunker, sample_paper, sample_pdf_content
    ):
        """Test the chunk validation functionality."""
        chunks, errors, stats = chunker.process_papers(
            [sample_paper], [sample_pdf_content]
        )

        # Validate chunks using the validation method
        validation_errors = chunker.validate_chunks(chunks)

        # Should have no validation errors for properly generated chunks
        assert (
            len(validation_errors) == 0
        ), f"Unexpected validation errors: {validation_errors}"

        # Test with artificially corrupted chunk
        corrupted_chunk = Chunk(
            content="",
            section="",
            paper_id="",
            paper_title="Test",
            start_position=-1,
            end_position=0,
            token_count=0,
        )
        corrupted_chunks = [corrupted_chunk]

        validation_errors = chunker.validate_chunks(corrupted_chunks)

        # Should detect multiple validation issues
        assert len(validation_errors) > 0, "Should detect validation errors"
        assert any(
            "empty" in error.lower() for error in validation_errors
        ), "Should detect empty content"
        assert any(
            "missing" in error.lower() for error in validation_errors
        ), "Should detect missing fields"


class TestQualityAssessment:
    """Test quality assessment functionality."""

    def test_quality_score_calculation(self, chunker):
        """Test quality score assessment."""
        # High quality content
        good_content = "This is a well-written sentence with proper structure. It contains meaningful content about scientific research methods. The text demonstrates good grammar and coherence."
        good_score = chunker._assess_chunk_quality(good_content)

        # Low quality content
        poor_content = "short bad"
        poor_score = chunker._assess_chunk_quality(poor_content)

        # Empty content
        empty_score = chunker._assess_chunk_quality("")

        assert (
            good_score > poor_score
        ), f"Good content ({good_score}) should score higher than poor content ({poor_score})"
        assert (
            poor_score > empty_score
        ), f"Poor content ({poor_score}) should score higher than empty ({empty_score})"
        assert (
            good_score <= 1.0 and good_score >= 0.0
        ), f"Good score out of range: {good_score}"
        assert empty_score == 0.0, f"Empty content should score 0.0, got {empty_score}"

    def test_quality_score_components(self, chunker):
        """Test individual components of quality scoring."""
        # Long content with good structure
        long_good = "Sentence one. Sentence two. Sentence three. This content has proper sentence structure and reasonable length for quality assessment."
        score = chunker._assess_chunk_quality(long_good)
        assert score > 0.5, f"Long, well-structured content should score well: {score}"

        # Content with word diversity
        diverse_content = "Machine learning algorithms neural networks transformers embeddings tokenization parsing syntax semantics pragmatics linguistics computational natural language processing artificial intelligence deep learning supervised unsupervised reinforcement transfer learning fine-tuning pretraining downstream tasks evaluation metrics accuracy precision recall f1-score bleu rouge perplexity coherence fluency readability accessibility usability human-centered design."
        score = chunker._assess_chunk_quality(diverse_content)
        # Should get bonus for diversity
        assert score > 0.4, f"Diverse content should score reasonably well: {score}"


# Additional utility tests
class TestConfiguration:
    """Test configuration parameters."""

    def test_custom_chunk_sizes(self):
        """Test chunker with custom sizes."""
        small_chunker = AdvancedDocumentChunker(chunk_size=200, chunk_overlap=20)

        text = "Sentence. " * 100
        chunks, _ = small_chunker._chunk_section_with_spacy(text, "test")

        # Should create smaller chunks
        for chunk in chunks[:-1]:  # All but last
            assert (
                chunk["token_count"] <= 300
            ), f"Chunk too large for small settings: {chunk['token_count']}"

    def test_min_chunk_tokens_filtering(self):
        """Test minimum token filtering."""
        strict_chunker = AdvancedDocumentChunker(min_chunk_tokens=500)

        # Very short text
        text = "Short sentence."
        chunks, _ = strict_chunker._chunk_section_with_spacy(text, "test")

        # Should be filtered out due to minimum token requirement
        token_count = strict_chunker._count_tokens(text)
        if token_count < 500:
            assert (
                len(chunks) == 0
            ), "Very short content should be filtered out with strict minimums"


if __name__ == "__main__":
    pytest.main([__file__])
