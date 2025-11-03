import re
from typing import List, Set, Dict, Any
from models.state import State, Paper
from loguru import logger


class DeduplicationAgent:
    """
    Deduplicates papers by DOI or normalized title.

    Handles cumulative paper pools from multiple search attempts during
    query refinement loops. Uses DOI as primary deduplication key (most reliable),
    falls back to normalized title for preprints without DOIs.

    Typical deduplication rates:
    - After 1 refinement (200 papers): ~25-30% duplicates
    - After 2 refinements (300 papers): ~30-40% duplicates
    """

    def deduplicate(self, state: State) -> State:
        """
        Deduplicate papers in state.papers_metadata.

        Args:
            state: LangGraph State with papers_metadata list

        Returns:
            Updated State with deduplicated papers_metadata and dedup_stats
        """
        raw_papers = state.papers_metadata
        raw_count = len(raw_papers)

        if raw_count == 0:
            logger.info("DeduplicationAgent: No papers to deduplicate")
            state.dedup_stats = {
                "raw_count": 0,
                "unique_count": 0,
                "duplicates_removed": 0,
            }
            return state

        logger.info(f"DeduplicationAgent: Deduplicating {raw_count} papers...")

        # Deduplicate
        unique_papers, seen_identifiers = self._deduplicate_papers(raw_papers)
        unique_count = len(unique_papers)
        duplicates_removed = raw_count - unique_count

        # Update state
        state.papers_metadata = unique_papers
        state.dedup_stats = {
            "raw_count": raw_count,
            "unique_count": unique_count,
            "duplicates_removed": duplicates_removed,
            "deduplication_rate": (
                duplicates_removed / raw_count if raw_count > 0 else 0
            ),
        }

        logger.info(
            f"DeduplicationAgent: {raw_count} → {unique_count} papers "
            f"({duplicates_removed} duplicates removed, {state.dedup_stats['deduplication_rate']:.1%} rate)"
        )

        return state

    def _deduplicate_papers(self, papers: List[Paper]) -> tuple[List[Paper], Set[str]]:
        """
        Deduplicate papers using DOI (primary) or normalized title (fallback).

        Algorithm:
        1. For each paper, generate identifier:
           - If DOI exists: normalize DOI (lowercase, strip https://doi.org/)
           - Else: normalize title (lowercase, strip punctuation, trim whitespace)
        2. Keep first occurrence of each identifier
        3. Track seen identifiers to detect duplicates

        Args:
            papers: List of Paper objects (may contain duplicates)

        Returns:
            Tuple of (unique_papers, seen_identifiers_set)
        """
        seen_identifiers: Set[str] = set()
        unique_papers: List[Paper] = []

        for paper in papers:
            # Generate deduplication identifier
            identifier = self._generate_identifier(paper)

            # Skip if already seen
            if identifier in seen_identifiers:
                logger.debug(f"Duplicate found: {paper.title[:50]}...")
                continue

            # Add to unique set
            seen_identifiers.add(identifier)
            unique_papers.append(paper)

        return unique_papers, seen_identifiers

    def _generate_identifier(self, paper: Paper) -> str:
        """
        Generate unique identifier for a paper.

        Priority:
        1. DOI (if present) — most reliable, globally unique
        2. Normalized title (fallback) — less reliable but works for preprints

        Args:
            paper: Paper object

        Returns:
            Normalized identifier string
        """
        # Priority 1: Use DOI if available
        if paper.doi:
            return self._normalize_doi(paper.doi)

        # Priority 2: Use paper_id if it's an OpenAlex ID
        if paper.paper_id and paper.paper_id.startswith("https://openalex.org/"):
            return paper.paper_id.lower()

        # Fallback: Use normalized title
        return self._normalize_title(paper.title)

    def _normalize_doi(self, doi: str) -> str:
        """
        Normalize DOI for deduplication.

        Handles variations:
        - https://doi.org/10.1234/example → 10.1234/example
        - DOI: 10.1234/example → 10.1234/example
        - 10.1234/EXAMPLE → 10.1234/example (lowercase)

        Args:
            doi: DOI string (may have prefixes or mixed case)

        Returns:
            Normalized DOI string
        """
        # Remove common prefixes
        doi = doi.lower().strip()
        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://dx.doi.org/", "")
        doi = doi.replace("doi:", "").strip()

        return doi

    def _normalize_title(self, title: str) -> str:
        """
        Normalize title for deduplication.

        Normalization steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove extra whitespace
        4. Strip leading/trailing whitespace

        Args:
            title: Paper title

        Returns:
            Normalized title string
        """
        if not title:
            return ""

        # Lowercase
        normalized = title.lower()

        # Remove punctuation (keep alphanumeric and spaces)
        normalized = re.sub(r"[^\w\s]", "", normalized)

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized
