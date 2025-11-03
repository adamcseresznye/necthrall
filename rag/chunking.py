"""
Advanced Document Chunking Module for Necthrall RAG System

This module implements intelligent, section-aware document chunking for scientific research papers.
Features token-based chunking, sentence boundary preservation, and intelligent overlap generation
using regex-based NLP processing.

Key Features:
- ✅ Token-based chunking (500 tokens default, 50 token overlap)
- ✅ Regex-based sentence boundary preservation
- ✅ Section-aware intelligent overlap
- ✅ Comprehensive metadata with token counts
- ✅ Memory-efficient processing for large documents
- ✅ Quality validation and error handling
- ✅ Configurable section pattern detection
- ✅ Performance optimized for < 1 second per 25 documents

Tech Stack: Python 3.11+, regex for NLP processing, regex for section detection
Fallback Note: Originally designed with SpaCy integration, now uses regex-based tokenization
to avoid Python 3.13 compatibility issues.
"""

from loguru import logger
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.state import Chunk, PDFContent
from utils.spacy_error_handler import SpaCyErrorHandler


class SectionPatternConfig:
    """Configuration for academic paper section pattern detection."""

    # Standard academic section patterns (case-insensitive, multi-format support)
    PATTERNS = {
        "abstract": re.compile(r"(?mi)^(\d+\.\s*)?(abstract|summary)\b"),
        "introduction": re.compile(
            r"(?mi)^(\d+\.\s*)?(introduction|intro|background)\b"
        ),
        "methods": re.compile(
            r"(?mi)^(\d+\.\s*)?(methods?|materials?\s+(and\s+)?methods?|procedure|methodology|experimental)\b"
        ),
        "results": re.compile(
            r"(?mi)^(\d+\.\s*)?(results?|findings?|experimental\s+results?|data)\b"
        ),
        "discussion": re.compile(
            r"(?mi)^(\d+\.\s*)?(discussion|discussion\s+(and\s+)?results?|interpretation|analysis)\b"
        ),
        "conclusion": re.compile(
            r"(?mi)^(\d+\.\s*)?(conclusion|conclusions?|summary|outlook|future\s+work)\b"
        ),
    }

    # Excluded patterns (subsections, appendices, references)
    EXCLUDE_PATTERNS = [
        re.compile(r"(?mi)^\d+\.\d+\s"),  # Subsection patterns (2.1, 3.2)
        re.compile(r"(?mi)^(appendix|supplemental|references?|bibliography)\b"),
        re.compile(r"(?mi)^(figure|table)\s+\d+"),  # Figure/table captions
    ]


class AdvancedDocumentChunker:
    """
    Advanced section-aware document chunker for scientific papers with regex-based NLP.

    Implements intelligent token-based chunking with sentence boundary preservation,
    configurable section detection, and memory-efficient processing.

    Core Features:
    - Token-based chunking (500 tokens target, 50 token intelligent overlap)
    - Regex-based sentence segmentation for natural language boundaries
    - Section-aware chunking with intelligent overlap
    - Comprehensive metadata with provenance information
    - Memory-efficient processing for large documents (50+ pages)
    - Quality validation and edge case handling
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_section_chars: int = 50,
        min_chunk_tokens: int = 50,
        memory_threshold_mb: int = 100,
        enable_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        enable_parallel: bool = False,
        max_workers: int = 4,
    ):
        """
        Initialize AdvancedDocumentChunker with token-based chunking and spaCy integration.

        Args:
            chunk_size: Target chunk size in tokens (default: 500)
            chunk_overlap: Minimum overlap between chunks in tokens (default: 50)
            min_section_chars: Minimum section length in characters to be processed (default: 50)
            min_chunk_tokens: Minimum tokens per chunk for quality filtering (default: 100)
            memory_threshold_mb: Memory usage threshold for processing large documents (default: 100MB)
            enable_spacy: Whether to attempt spaCy integration (default: True)
            spacy_model: spaCy model to use (default: en_core_web_sm)
            enable_parallel: Enable parallel processing (default: False)
            max_workers: Maximum worker threads for parallel processing (default: 4)
        """
        # Configuration parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_chars = min_section_chars
        self.min_chunk_tokens = min_chunk_tokens
        self.memory_threshold_mb = memory_threshold_mb
        self.enable_spacy = enable_spacy
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

        # Initialize SpaCy error handler
        self.spacy_handler = None
        if enable_spacy:
            try:
                self.spacy_handler = SpaCyErrorHandler(
                    model_name=spacy_model, enable_logging=True
                )
                # Attempt to load the model
                nlp = self.spacy_handler.load_model()
                if nlp is None:
                    logger.warning(
                        "spaCy model failed to load, will use regex fallbacks"
                    )
            except Exception as e:
                logger.warning(f"SpaCy handler initialization failed: {e}")
                self.spacy_handler = None

        # Fallback tokenization pattern (regex-based)
        self.token_pattern = re.compile(r"\b\w+\b")
        self.regex_tokenizer = None

        # Section detection configuration
        self.section_config = SectionPatternConfig()

        logger.info(
            f"AdvancedDocumentChunker initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, spacy_enabled={enable_spacy}, "
            f"parallel_enabled={enable_parallel}"
        )

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using spaCy if available, fallback to regex."""
        if self.spacy_handler and self.spacy_handler.nlp:
            try:
                tokens = self.spacy_handler.process_text(text, operation="tokenize")
                if tokens:
                    return tokens
            except Exception as e:
                logger.debug(f"spaCy tokenization failed: {e}")

        # Fallback to regex tokenization
        if self.regex_tokenizer is None and self.spacy_handler:
            self.regex_tokenizer = self.spacy_handler.get_regex_tokenizer()

        if self.regex_tokenizer:
            return self.regex_tokenizer["tokenize"](text)
        else:
            # Ultimate fallback to simple regex
            return self.token_pattern.findall(text)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using available tokenization method."""
        return len(self._tokenize_text(text))

    def _split_sentences(self, text: str) -> List[str]:
        """Split sentences using spaCy if available, fallback to regex."""
        if self.spacy_handler and self.spacy_handler.nlp:
            try:
                sentences = self.spacy_handler.process_text(
                    text, operation="sentencize"
                )
                if sentences:
                    return [
                        str(sent).strip() for sent in sentences if str(sent).strip()
                    ]
            except Exception as e:
                logger.debug(f"spaCy sentence splitting failed: {e}")

        # Fallback to regex sentence splitting
        if self.regex_tokenizer is None and self.spacy_handler:
            self.regex_tokenizer = self.spacy_handler.get_regex_tokenizer()

        if self.regex_tokenizer:
            return self.regex_tokenizer["split_sentences"](text)
        else:
            # Ultimate fallback to simple regex
            sentence_pattern = re.compile(r"(?<=[.!?])\s+")
            sentences = sentence_pattern.split(text.strip())
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

            # Fallback: if no sentences found, treat whole text as one sentence
            if not sentences:
                sentences = [text.strip()]

            return sentences

    def process_papers(
        self,
        papers: List[Any],
        pdf_contents: List[PDFContent],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Chunk], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process multiple papers into token-based chunks with comprehensive metadata.

        Args:
            papers: List of paper objects with paper_id and title attributes
            pdf_contents: List of PDFContent objects with raw_text
            config: Optional configuration override (chunk_size, overlap, section_patterns)

        Returns:
            Tuple of (chunks, errors, performance_stats)

        Performance Requirements:
        - Process 25 documents in < 1 second
        - Generate 50-100 chunks per typical academic paper
        - Handle documents up to 50 pages without degradation
        """
        start_time = time.time()
        all_chunks = []
        all_errors = []

        # Apply configuration overrides
        if config:
            self.chunk_size = config.get("chunk_size", self.chunk_size)
            self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
            if "section_patterns" in config:
                self.section_config.PATTERNS.update(config["section_patterns"])

        # Build PDF content lookup for efficient processing
        pdf_lookup = {pdf.paper_id: pdf for pdf in pdf_contents}
        total_papers = len(papers)

        # Use parallel processing if enabled and we have multiple papers
        if self.enable_parallel and len(papers) > 1:
            all_chunks, all_errors, paper_metrics = self._process_papers_parallel(
                papers, pdf_lookup, total_papers
            )
            processed_papers = paper_metrics["processed"]
            skipped_papers = paper_metrics["skipped"]
            total_tokens = paper_metrics["tokens"]
        else:
            # Sequential processing
            processed_papers = 0
            skipped_papers = 0
            total_tokens = 0

            for paper in papers:
                paper_chunks, paper_errors, paper_tokens = (
                    self._process_single_paper_wrapper(paper, pdf_lookup, total_papers)
                )
                all_chunks.extend(paper_chunks)
                all_errors.extend(paper_errors)
                total_tokens += paper_tokens

                if paper_chunks:
                    processed_papers += 1
                else:
                    skipped_papers += 1

        # Calculate performance metrics
        total_time = time.time() - start_time
        throughput_chunks_per_second = (
            len(all_chunks) / total_time if total_time > 0 else 0
        )

        # Calculate section distribution
        section_counts = {}
        for chunk in all_chunks:
            section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1

        performance_stats = {
            "total_time": total_time,
            "processed_papers": processed_papers,
            "total_papers": total_papers,
            "skipped_papers": skipped_papers,
            "total_chunks": len(all_chunks),
            "total_tokens": total_tokens,
            "throughput_chunks_per_second": throughput_chunks_per_second,
            "avg_chunks_per_paper": len(all_chunks) / max(processed_papers, 1),
            "section_distribution": section_counts,
            "total_errors": len(all_errors),
            "parallel_processing": self.enable_parallel,
            "workers_used": self.max_workers if self.enable_parallel else 1,
        }

        logger.info(
            ".3f"
            f"{processed_papers}/{total_papers} papers, "
            ".1f"
            f"{len(all_errors)} errors, parallel={self.enable_parallel}"
        )

        return all_chunks, all_errors, performance_stats

    def _process_papers_parallel(
        self, papers: List[Any], pdf_lookup: Dict[str, PDFContent], total_papers: int
    ) -> Tuple[List[Chunk], List[Dict[str, Any]], Dict[str, int]]:
        """
        Process papers in parallel using thread pool.

        Args:
            papers: List of paper objects
            pdf_lookup: PDF content lookup dictionary
            total_papers: Total number of papers for logging

        Returns:
            Tuple of (chunks, errors, metrics)
        """
        all_chunks = []
        all_errors = []
        processed_papers = 0
        total_tokens = 0

        logger.info(
            f"Processing {len(papers)} papers in parallel with {self.max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all paper processing tasks
            future_to_paper = {
                executor.submit(
                    self._process_single_paper_wrapper, paper, pdf_lookup, total_papers
                ): paper
                for paper in papers
            }

            # Collect results as they complete
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    paper_chunks, paper_errors, paper_tokens = future.result()

                    all_chunks.extend(paper_chunks)
                    all_errors.extend(paper_errors)
                    total_tokens += paper_tokens

                    if paper_chunks:
                        processed_papers += 1
                    else:
                        logger.debug(f"Paper {paper.paper_id} produced no chunks")

                except Exception as e:
                    # Handle unexpected errors in parallel processing
                    error_info = {
                        "paper_id": getattr(paper, "paper_id", "unknown"),
                        "stage": "parallel_processing",
                        "error_type": type(e).__name__,
                        "message": f"Parallel processing failed: {str(e)}",
                        "timestamp": time.time(),
                    }
                    all_errors.append(error_info)
                    logger.error(
                        f"Parallel processing error for paper {paper.paper_id}: {e}"
                    )

        skipped_papers = (
            total_papers
            - processed_papers
            - len(
                [e for e in all_errors if e.get("stage") in ["pdf_lookup", "general"]]
            )
        )

        return (
            all_chunks,
            all_errors,
            {
                "processed": processed_papers,
                "skipped": skipped_papers,
                "tokens": total_tokens,
            },
        )

    def _process_single_paper_wrapper(
        self, paper: Any, pdf_lookup: Dict[str, PDFContent], total_papers: int
    ) -> Tuple[List[Chunk], List[Dict[str, Any]], int]:
        """
        Wrapper for single paper processing with uniform error handling.

        Args:
            paper: Paper object to process
            pdf_lookup: PDF content lookup dictionary
            total_papers: Total papers for logging context

        Returns:
            Tuple of (chunks, errors, tokens)
        """
        try:
            # Validate input data
            if not hasattr(paper, "paper_id") or not hasattr(paper, "title"):
                raise ValueError(
                    "Paper object missing required attributes: paper_id, title"
                )

            paper_id = paper.paper_id

            # Find corresponding PDF content
            if paper_id not in pdf_lookup:
                error_info = {
                    "paper_id": paper_id,
                    "stage": "pdf_lookup",
                    "error_type": "MissingPDFContent",
                    "message": f"No PDF content found for paper {paper_id}",
                    "timestamp": time.time(),
                }
                return [], [error_info], 0

            pdf_content = pdf_lookup[paper_id]

            # Process single paper
            paper_chunks, paper_errors, paper_tokens = self._process_single_paper(
                paper, pdf_content
            )

            return paper_chunks, paper_errors, paper_tokens

        except Exception as e:
            error_info = {
                "paper_id": getattr(paper, "paper_id", "unknown"),
                "stage": "general",
                "error_type": type(e).__name__,
                "message": f"Unexpected error processing paper: {str(e)}",
                "timestamp": time.time(),
            }
            logger.error(f"AdvancedDocumentChunker: {error_info['message']}")
            return [], [error_info], 0

    def _process_single_paper(
        self, paper: Any, pdf_content: PDFContent
    ) -> Tuple[List[Chunk], List[Dict[str, Any]], int]:
        """
        Process a single paper into chunks with section detection and token-based chunking.

        Args:
            paper: Paper object with paper_id and title
            pdf_content: PDFContent object with raw_text

        Returns:
            Tuple of (chunks, errors, total_tokens_processed)
        """
        chunks = []
        errors = []
        total_tokens = 0

        try:
            # Input validation
            self._validate_input(paper, pdf_content)

            # Detect sections using regex patterns
            sections = self._detect_sections_with_metadata(pdf_content.raw_text)

            # Determine if we should use section-aware or fallback chunking
            use_fallback = len(sections) < 2 or all(  # Not enough sections detected
                len(section["content"]) < self.min_section_chars for section in sections
            )  # All sections too short

            if use_fallback:
                logger.warning(
                    f"Paper {paper.paper_id}: Using fallback chunking "
                    f"(detected sections: {len(sections)})"
                )
                paper_chunks, chunk_tokens = self._chunk_text_fallback_with_spacy(
                    pdf_content.raw_text
                )
                total_tokens += chunk_tokens

                for chunk_data in paper_chunks:
                    chunk = Chunk(
                        content=chunk_data["content"],
                        section="unknown",
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        start_position=chunk_data["start_pos"],
                        end_position=chunk_data["end_pos"],
                        use_fallback=True,
                        token_count=chunk_data["token_count"],
                        metadata={
                            "chunk_index": len(chunks),
                            "chunking_strategy": "fallback_paragraph",
                            "quality_score": self._assess_chunk_quality(
                                chunk_data["content"]
                            ),
                        },
                    )
                    chunks.append(chunk)
            else:
                # Section-aware chunking with intelligent overlap
                for section_idx, section in enumerate(sections):
                    if len(section["content"]) < self.min_section_chars:
                        logger.info(
                            f"Skipping short section '{section['section']}' ({len(section['content'])} chars)"
                        )
                        continue

                    section_chunks, section_tokens = self._chunk_section_with_spacy(
                        section["content"], section["section"]
                    )
                    total_tokens += section_tokens

                    for chunk_data in section_chunks:
                        chunk = Chunk(
                            content=chunk_data["content"],
                            section=section["section"],
                            paper_id=paper.paper_id,
                            paper_title=paper.title,
                            start_position=chunk_data["start_pos"]
                            + section["start_pos"],
                            end_position=chunk_data["end_pos"] + section["start_pos"],
                            use_fallback=False,
                            token_count=chunk_data["token_count"],
                            metadata={
                                "chunk_index": len(chunks),
                                "section_index": section_idx,
                                "chunking_strategy": "section_aware_intelligent_overlap",
                                "quality_score": self._assess_chunk_quality(
                                    chunk_data["content"]
                                ),
                                "overlap_preserved": chunk_data.get(
                                    "overlap_preserved", False
                                ),
                            },
                        )
                        chunks.append(chunk)

        except Exception as e:
            error_info = {
                "paper_id": paper.paper_id,
                "stage": "chunking",
                "error_type": type(e).__name__,
                "message": f"Chunking failed: {str(e)}",
                "timestamp": time.time(),
            }
            errors.append(error_info)
            logger.error(f"AdvancedDocumentChunker: {error_info['message']}")

        return chunks, errors, total_tokens

    def _validate_input(self, paper: Any, pdf_content: PDFContent) -> None:
        """Validate input parameters and data integrity."""
        if not pdf_content.raw_text or not pdf_content.raw_text.strip():
            raise ValueError(
                f"Empty or whitespace-only text for paper {paper.paper_id}"
            )

        # Check for obviously corrupted text
        text_bytes = len(pdf_content.raw_text.encode("utf-8", errors="replace"))
        if text_bytes > self.memory_threshold_mb * 1024 * 1024:
            raise ValueError(
                f"Text size {text_bytes} bytes exceeds memory threshold "
                f"{self.memory_threshold_mb}MB for paper {paper.paper_id}"
            )

        # Check for excessive null bytes (corruption indicator)
        if (
            "\x00" in pdf_content.raw_text
            and pdf_content.raw_text.count("\x00") > len(pdf_content.raw_text) * 0.01
        ):
            raise ValueError(
                f"Text appears corrupted (excessive null bytes) for paper {paper.paper_id}"
            )

    def _detect_sections_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect academic paper sections using comprehensive regex patterns.

        Args:
            text: Full document text

        Returns:
            List of section dictionaries with content, section type, and metadata
        """
        sections = []

        # Find all section matches with positions
        section_positions = []
        for section_name, pattern in self.section_config.PATTERNS.items():
            for match in pattern.finditer(text):
                section_positions.append(
                    {
                        "name": section_name,
                        "start": match.start(),
                        "header_end": match.end(),
                        "match": match,
                    }
                )

        # Sort by position and filter overlapping detections
        section_positions.sort(key=lambda x: x["start"])
        filtered_positions = self._filter_overlapping_sections(section_positions, text)

        # Extract section content between detected headers
        for i, pos in enumerate(filtered_positions):
            start_pos = pos["start"]
            end_pos = (
                filtered_positions[i + 1]["start"]
                if i + 1 < len(filtered_positions)
                else len(text)
            )

            # Extract clean section content
            section_content = text[start_pos:end_pos]
            sections.append(
                {
                    "section": pos["name"],
                    "content": section_content,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "header_length": pos["header_end"] - pos["start"],
                }
            )

        return sections

    def _filter_overlapping_sections(
        self, positions: List[Dict], text: str
    ) -> List[Dict]:
        """Filter overlapping section detections, preferring main sections."""
        if not positions:
            return positions

        # Remove subsections and appendices first
        filtered = []
        for pos in positions:
            header_text = text[pos["start"] : pos["header_end"]].lower()
            skip_section = False

            for exclude_pattern in self.section_config.EXCLUDE_PATTERNS:
                if exclude_pattern.search(header_text):
                    skip_section = True
                    break

            if not skip_section:
                filtered.append(pos)

        # Remove position overlaps (prefer exact matches earlier in document)
        filtered.sort(key=lambda x: x["start"])
        deduplicated = []
        prev_end = -1

        for pos in filtered:
            if pos["start"] >= prev_end:
                deduplicated.append(pos)
                prev_end = pos["header_end"]

        return deduplicated

    def _chunk_section_with_spacy(
        self, section_text: str, section_name: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Perform intelligent token-based chunking with sentence boundary preservation.

        Args:
            section_text: Section content to chunk
            section_name: Section name for logging

        Returns:
            Tuple of (chunk_list, total_tokens)
        """
        if not section_text or not section_text.strip():
            return [], 0

        # Simple sentence segmentation
        sentences = self._split_sentences(section_text)

        if not sentences:
            # Fallback if no sentences detected
            logger.warning(
                f"No sentences detected in section '{section_name}', using paragraph fallback"
            )
            return self._chunk_text_fallback_with_spacy(section_text)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        start_pos = 0

        for sent_idx, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # End current chunk
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "start_pos": start_pos,
                        "end_pos": start_pos + len(current_chunk),
                        "token_count": current_tokens,
                        "overlap_preserved": len(chunks)
                        > 0,  # All chunks after first have overlap
                    }
                )

                # Start new chunk with intelligent overlap
                overlap_start = self._find_overlap_start(
                    sentences, sent_idx, self.chunk_overlap
                )
                overlap_sentences = sentences[overlap_start : sent_idx + 1]

                current_chunk = " ".join(overlap_sentences)
                current_tokens = self._count_tokens(current_chunk)
                start_pos = (
                    sent_idx - len(overlap_sentences) + 1
                )  # Approximate character position
                start_pos = max(start_pos, 0)

            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

            current_tokens += sentence_tokens

        # Add final chunk if not empty
        if current_chunk and current_tokens >= self.min_chunk_tokens:
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(current_chunk),
                    "token_count": current_tokens,
                    "overlap_preserved": len(chunks) > 0,
                }
            )

        total_tokens = sum(self._count_tokens(chunk["content"]) for chunk in chunks)
        return chunks, total_tokens

    def _chunk_text_fallback_with_spacy(
        self, text: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Fallback chunking using paragraph boundaries when section detection fails.

        Args:
            text: Full text to chunk

        Returns:
            Tuple of (chunk_list, total_tokens)
        """
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return [], 0

        chunks = []
        current_chunk = ""
        current_tokens = 0
        start_pos = 0

        for para_idx, paragraph in enumerate(paragraphs):
            # Skip very short paragraphs (likely headers or artifacts)
            if len(paragraph) < 20:
                continue

            para_tokens = self._count_tokens(paragraph)

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # End current chunk
                if current_tokens >= self.min_chunk_tokens:
                    chunks.append(
                        {
                            "content": current_chunk.strip(),
                            "start_pos": start_pos,
                            "end_pos": start_pos + len(current_chunk),
                            "token_count": current_tokens,
                        }
                    )
                current_chunk = paragraph
                current_tokens = para_tokens
                start_pos = para_idx
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk and current_tokens >= self.min_chunk_tokens:
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(current_chunk),
                    "token_count": current_tokens,
                }
            )

        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        return chunks, total_tokens

    def _find_overlap_start(
        self, sentences: List[str], current_idx: int, target_overlap: int
    ) -> int:
        """
        Find optimal start position for overlap to achieve target token overlap.

        Args:
            sentences: List of sentence texts
            current_idx: Current sentence index
            target_overlap: Target overlap in tokens

        Returns:
            Start index for overlap sentences
        """
        overlap_tokens = 0
        overlap_start = current_idx

        # Work backwards from current sentence to accumulate target overlap tokens
        for idx in range(
            current_idx, max(-1, current_idx - 10), -1
        ):  # Check up to 10 sentences back
            sent_tokens = self._count_tokens(sentences[idx])
            overlap_tokens += sent_tokens
            overlap_start = idx

            if overlap_tokens >= target_overlap:
                break

        return overlap_start

    def _assess_chunk_quality(self, content: str) -> float:
        """
        Assess chunk quality based on content characteristics.

        Args:
            content: Chunk content to assess

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not content or not content.strip():
            return 0.0

        score = 0.05  # Small base score for any non-empty content

        # Length factor (prefer substantial content)
        content_length = len(content.strip())
        if content_length > 200:
            score += 0.4
        elif content_length > 100:
            score += 0.3
        elif content_length > 50:
            score += 0.2

        # Sentence structure (prefer natural language)
        sentence_count = content.count(".") + content.count("?") + content.count("!")
        if sentence_count > 2:
            score += 0.3
        elif sentence_count > 0:
            score += 0.2

        # Word diversity (prefer meaningful content)
        words = [word for word in content.split() if word.isalpha()]
        if len(words) > 20:
            unique_words = len(set(word.lower() for word in words))
            diversity = unique_words / len(words)
            score += min(diversity * 0.3, 0.3)

        return min(score, 1.0)

    def validate_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Perform comprehensive validation of chunk quality and metadata.

        Args:
            chunks: List of Chunk objects to validate

        Returns:
            List of validation error messages
        """
        errors = []

        for i, chunk in enumerate(chunks):
            # Content validation
            if not chunk.content or not chunk.content.strip():
                errors.append(f"Chunk {i}: Empty or whitespace-only content")
                # Continue to check other fields even if content is empty

            # Token count validation
            if chunk.token_count < self.min_chunk_tokens:
                errors.append(
                    f"Chunk {i}: Token count too low ({chunk.token_count} < {self.min_chunk_tokens})"
                )

            # Metadata validation
            if not chunk.paper_id:
                errors.append(f"Chunk {i}: Missing paper_id")
            if not chunk.section:
                errors.append(f"Chunk {i}: Missing section")

            # Position validation
            if chunk.start_position < 0 or chunk.end_position <= chunk.start_position:
                errors.append(
                    f"Chunk {i}: Invalid position range [{chunk.start_position}, {chunk.end_position}]"
                )

            # Content consistency
            if len(chunk.content) > (chunk.end_position - chunk.start_position) * 2:
                errors.append(
                    f"Chunk {i}: Content length ({len(chunk.content)}) inconsistent with position"
                )

            # Quality check
            quality = self._assess_chunk_quality(chunk.content)
            if quality < 0.3:
                errors.append(
                    f"Chunk {i}: Low quality score ({quality:.2f}), may be metadata or artifacts"
                )

        return errors

    def get_performance_stats(
        self, chunks: List[Chunk], processing_time: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance and quality statistics.

        Args:
            chunks: List of processed chunks
            processing_time: Total processing time in seconds

        Returns:
            Dictionary with performance metrics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "processing_time": processing_time,
                "avg_chunk_length": 0,
                "quality_distribution": {},
            }

        # Calculate chunk statistics
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_chunk_length = total_chars / len(chunks)

        # Section distribution
        section_counts = {}
        quality_scores = []
        for chunk in chunks:
            section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1
            quality_scores.append(self._assess_chunk_quality(chunk.content))

        # Quality distribution
        quality_distribution = {
            "excellent": sum(1 for score in quality_scores if score >= 0.8),
            "good": sum(1 for score in quality_scores if 0.6 <= score < 0.8),
            "acceptable": sum(1 for score in quality_scores if 0.4 <= score < 0.6),
            "poor": sum(1 for score in quality_scores if score < 0.4),
        }

        return {
            "total_chunks": len(chunks),
            "processing_time": processing_time,
            "avg_chunk_length": avg_chunk_length,
            "section_distribution": section_counts,
            "quality_distribution": quality_distribution,
            "avg_quality_score": (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            ),
            "throughput_chars_per_second": (
                total_chars / processing_time if processing_time > 0 else 0
            ),
        }
