import re
import time
from typing import List, Literal, Optional, TypedDict, Dict
from loguru import logger
import json


# Define output structure using TypedDict for simple data structures
class SectionChunk(TypedDict):
    content: str
    section: Literal[
        "introduction", "methods", "results", "discussion", "conclusion", "unknown"
    ]
    start_pos: int


class DetectionStats(TypedDict):
    total_sections: int
    sections_by_type: Dict[str, int]
    fallback_used: bool
    detection_success_rate: float
    processing_time_ms: float
    text_length: int
    memory_efficient_mode: bool


class SectionDetector:
    """
    Regex-based detector for academic paper sections.

    Detects standard academic paper sections (Introduction, Methods, Results,
    Discussion, Conclusion) and provides fallback chunking for poorly structured papers.

    Usage example:
        detector = SectionDetector()
        chunks = detector.detect_sections("1. Introduction\n\nText here...\n\n2. Methods\n\nMore text...")
        for chunk in chunks:
            print(f"{chunk['section']}: {chunk['content'][:50]}...")
    """

    def __init__(self, max_text_size_mb: int = 100):
        # Performance and safety limits
        self.max_text_size = max_text_size_mb * 1024 * 1024  # Convert MB to bytes
        self.timeout_seconds = 30  # Maximum processing time
        self.memory_efficient_threshold = (
            50 * 1024 * 1024
        )  # 50MB threshold for memory-efficient mode

        # Section detection patterns (case-insensitive, handles numbered and unnumbered headers)
        self.section_patterns = {
            "introduction": re.compile(
                r"(?mi)^(\d+\.\s*)?(introduction|intro|abstract)\b"
            ),
            "methods": re.compile(
                r"(?mi)^(\d+\.\s*)?(methods?|materials?\s+and\s+methods?|procedure|experimental)\b"
            ),
            "results": re.compile(
                r"(?mi)^(\d+\.\s*)?(results?|findings?|experimental\s+results?)\b"
            ),
            "discussion": re.compile(
                r"(?mi)^(\d+\.\s*)?(discussion|discussion\s+and\s+results?)\b"
            ),
            "conclusion": re.compile(
                r"(?mi)^(\d+\.\s*)?(conclusion|summary|outlook|conclusions)\b"
            ),
        }

        # Patterns to exclude (subsections, appendices, etc.)
        self.exclude_patterns = [
            re.compile(r"(?mi)^\d+\.\d+\s"),  # Subsection patterns like "2.1", "3.2"
            re.compile(
                r"(?mi)^(appendix|supplemental|references?)\b"
            ),  # Common non-main sections
        ]

        # Fallback chunking parameters
        self.chunk_size = 1000  # Character size for fallback chunks
        self.chunk_overlap = 100  # Overlap between chunks

        # Standard academic sections for success rate calculation
        self.standard_sections = {
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        }

    def detect_sections(self, text: str) -> List[SectionChunk]:
        """
        Detect and extract academic paper sections from text with comprehensive error handling and performance optimizations.

        Args:
            text: Full paper text extracted from PDF

        Returns:
            List of SectionChunk dictionaries with content, section type, and position

        Raises:
            ValueError: If input text is invalid or exceeds size limits
            TimeoutError: If processing exceeds time limits
        """
        start_time = time.time()
        memory_efficient_mode = False

        # Enhanced input validation and error handling
        try:
            self._validate_input(text)
        except Exception as e:
            logger.warning(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input text: {e}") from e

        def timeout_handler():
            raise TimeoutError(
                f"Section detection exceeded {self.timeout_seconds}s timeout"
            )

        # Check if memory-efficient mode is needed
        memory_efficient_mode = len(text) > self.memory_efficient_threshold

        try:
            sections = []
            section_positions = []

            # Find all section matches efficiently
            if memory_efficient_mode:
                section_positions = self._find_sections_memory_efficient(text)
            else:
                section_positions = self._find_sections_standard(text)

            # Sort by position
            section_positions.sort(key=lambda x: x["start"])

            # Filter out excluded patterns (subsections, appendices, etc.)
            section_positions = self._filter_excluded_patterns(section_positions, text)

            # Remove overlapping detections (prefer earlier match if positions overlap)
            filtered_positions = self._remove_overlapping_positions(section_positions)

            if len(filtered_positions) >= 2:
                # Extract sections with boundaries between detected headers
                sections = self._extract_section_content(filtered_positions, text)
            else:
                # Fallback to chunking if fewer than 2 sections detected
                sections = self._chunk_text(text)

            # Filter out empty or minimal sections after extraction
            sections = self._filter_empty_sections(sections)

            execution_time = time.time() - start_time

            # Calculate comprehensive statistics
            stats = self._calculate_detection_stats(
                sections,
                filtered_positions,
                text,
                execution_time,
                memory_efficient_mode,
            )

            # Structured logging with detailed statistics
            logger.info(json.dumps(stats))

            return sections

        except TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(
                "Section detection timed out",
                json.dumps(
                    {
                        "event": "section_detection_timeout",
                        "text_length": len(text),
                        "execution_time_ms": round(execution_time * 1000, 2),
                    }
                ),
            )
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Section detection failed",
                json.dumps(
                    {
                        "event": "section_detection_error",
                        "error": str(e),
                        "text_length": len(text),
                        "execution_time_ms": round(execution_time * 1000, 2),
                    }
                ),
            )
            raise RuntimeError(f"Section detection failed: {e}") from e

    def _validate_input(self, text: str) -> None:
        """Comprehensive input validation with error handling for malformed PDFs."""
        if not isinstance(text, str):
            raise ValueError(
                "Input must be a string, not {}".format(type(text).__name__)
            )

        # Check size limits
        text_bytes = len(text.encode("utf-8", errors="replace"))
        if text_bytes > self.max_text_size:
            raise ValueError(
                f"Text size {text_bytes} bytes exceeds maximum allowed size of {self.max_text_size} bytes"
            )

        # Strip and check for meaningful content
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text is empty or contains only whitespace")

        # Check for obviously corrupted text (too many null bytes, etc.)
        if (
            "\x00" in text and text.count("\x00") > len(text) * 0.01
        ):  # More than 1% null bytes
            raise ValueError(
                "Input text appears corrupted (contains excessive null bytes)"
            )

        # Check encoding issues - try to encode/decode to catch encoding problems
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as e:
            raise ValueError(f"Input text contains invalid UTF-8 characters: {e}")

    def _find_sections_standard(self, text: str) -> List[Dict]:
        """Standard section detection for normal-sized texts."""
        section_positions = []

        # Find all section matches and their positions
        for section_name, pattern in self.section_patterns.items():
            for match in pattern.finditer(text):
                section_positions.append(
                    {
                        "name": section_name,
                        "start": match.start(),
                        "header_end": match.end(),
                        "match": match,
                    }
                )

        return section_positions

    def _find_sections_memory_efficient(self, text: str) -> List[Dict]:
        """Memory-efficient section detection for large texts using chunking."""
        section_positions = []
        text_len = len(text)
        chunk_size = min(
            1024 * 1024, text_len // 4
        )  # Process in 1MB chunks or quarters

        for start in range(0, text_len, chunk_size):
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]

            # Add overlap to catch sections split across chunk boundaries
            overlap_start = max(0, start - 500)
            overlap_end = min(text_len, end + 500)

            if overlap_start < start:
                chunk = text[overlap_start:end]
            elif overlap_end > end:
                chunk = text[start:overlap_end]

            # Find sections in this chunk
            chunk_offset = overlap_start if overlap_start < start else start

            for section_name, pattern in self.section_patterns.items():
                for match in pattern.finditer(chunk):
                    global_start = chunk_offset + match.start()

                    # Skip matches that would be outside our original chunk (for boundary matches)
                    if global_start >= start and global_start < end:
                        section_positions.append(
                            {
                                "name": section_name,
                                "start": global_start,
                                "header_end": global_start
                                + (match.end() - match.start()),
                                "match": match,
                            }
                        )

        return section_positions

    def _filter_excluded_patterns(self, positions: List[Dict], text: str) -> List[Dict]:
        """Filter out section headers that match excluded patterns (subsections, appendices, etc.)."""
        filtered_positions = []

        for pos in positions:
            # Check if this position matches any exclude patterns
            is_excluded = False
            for exclude_pattern in self.exclude_patterns:
                header_text = text[pos["start"] : pos["header_end"]]
                if exclude_pattern.search(header_text):
                    is_excluded = True
                    break

            if not is_excluded:
                filtered_positions.append(pos)

        return filtered_positions

    def _remove_overlapping_positions(self, positions: List[Dict]) -> List[Dict]:
        """Remove overlapping section detections, preferring earlier matches."""
        if not positions:
            return positions

        positions.sort(key=lambda x: x["start"])
        filtered = []
        prev_end = -1

        for pos in positions:
            if pos["start"] >= prev_end:
                filtered.append(pos)
                prev_end = pos["header_end"]

        return filtered

    def _extract_section_content(
        self, positions: List[Dict], text: str
    ) -> List[SectionChunk]:
        """Extract section content between detected headers, filtering out excluded content."""
        sections = []

        for i, pos in enumerate(positions):
            start_pos = pos["start"]
            end_pos = positions[i + 1]["start"] if i + 1 < len(positions) else len(text)

            content = text[start_pos:end_pos]

            # Clean content by removing lines that match exclude patterns
            content_lines = content.split("\n")
            cleaned_lines = []

            for line in content_lines:
                # Check if this line starts with excluded patterns (after stripping)
                line_stripped = line.strip()
                skip_line = False

                for exclude_pattern in self.exclude_patterns:
                    if exclude_pattern.match(line_stripped):
                        skip_line = True
                        break

                if not skip_line:
                    cleaned_lines.append(line)

            cleaned_content = "\n".join(cleaned_lines)

            sections.append(
                {
                    "content": cleaned_content,
                    "section": pos["name"],
                    "start_pos": start_pos,
                }
            )

        return sections

    def _filter_empty_sections(
        self, sections: List[SectionChunk]
    ) -> List[SectionChunk]:
        """Filter out sections with insufficient meaningful content."""
        filtered_sections = []

        for section in sections:
            # Strip whitespace and check minimum length
            content = section["content"].strip()

            # Skip if content is too short
            if len(content) < 20:
                continue

            # Skip if content is mostly whitespace/non-printable
            printable_chars = sum(
                1 for c in content if c.isprintable() and not c.isspace()
            )
            if printable_chars < 10:
                continue

            # Skip if section contains only the header itself
            header_lines = [
                line.strip() for line in content.split("\n") if line.strip()
            ]
            if len(header_lines) < 2:  # Need at least header + some actual content
                continue

            # Update content and add to results
            section["content"] = content
            filtered_sections.append(section)

        return filtered_sections

    def _calculate_detection_stats(
        self,
        sections: List[SectionChunk],
        filtered_positions: List[Dict],
        text: str,
        execution_time: float,
        memory_efficient_mode: bool,
    ) -> DetectionStats:
        """Calculate comprehensive detection statistics."""
        # Count sections by type
        sections_by_type = {}
        for section in sections:
            section_type = section["section"]
            sections_by_type[section_type] = sections_by_type.get(section_type, 0) + 1

        # Calculate success rate (how many standard sections were found)
        found_standard_sections = set(sections_by_type.keys()) & self.standard_sections
        detection_success_rate = len(found_standard_sections) / len(
            self.standard_sections
        )

        return {
            "event": "section_detection_complete",
            "total_sections": len(sections),
            "sections_by_type": sections_by_type,
            "fallback_used": len(filtered_positions) < 2,
            "detection_success_rate": round(detection_success_rate, 3),
            "processing_time_ms": round(execution_time * 1000, 2),
            "text_length": len(text),
            "memory_efficient_mode": memory_efficient_mode,
            "standard_sections_found": list(found_standard_sections),
        }

    def _chunk_text(self, text: str) -> List[SectionChunk]:
        """
        Fallback method to split text into fixed-size chunks when section detection fails.

        Args:
            text: Full paper text

        Returns:
            List of SectionChunk dictionaries for chunked content
        """
        chunks = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Ensure we don't cut sentences mid-way if possible
            if end < text_len:
                # Look for sentence boundaries within last 200 chars
                last_period = max(
                    text.rfind(".", end - 200, end),
                    text.rfind("?", end - 200, end),
                    text.rfind("!", end - 200, end),
                )
                if last_period > end - 200:
                    end = last_period + 1

            content = text[start:end].strip()
            if (
                content and len(content) >= 10
            ):  # Only add non-empty chunks with minimum content
                chunks.append(
                    {"content": content, "section": "unknown", "start_pos": start}
                )

            # Move start position with overlap for next chunk
            start = max(start + self.chunk_size - self.chunk_overlap, end)

        return chunks
