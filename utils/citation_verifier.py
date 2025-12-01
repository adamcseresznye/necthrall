"""Citation verification utility for synthesized answers.

This module provides a CitationVerifier class to validate that inline citations
in synthesized answers (e.g., [1], [2]) correspond to valid passage indices.
"""

import re
from typing import Any, Dict, List

from loguru import logger


class CitationVerifier:
    """Verifies inline citations in synthesized answers.

    Validates that all [N] citation tags in an answer string reference valid
    passage indices (1 to len(passages)).

    Usage:
        >>> verifier = CitationVerifier()
        >>> result = verifier.verify("This is claimed [1] and supported [2].", passages)
        >>> print(result["valid"])
        True
    """

    # Regex pattern to match citation tags like [1], [2], [10], etc.
    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def verify(self, answer: str, passages: List[Any]) -> Dict[str, Any]:
        """Verify that all citations in the answer reference valid passages.

        Args:
            answer: The synthesized answer text containing [N] citation tags.
            passages: List of retrieved passages. Valid citation indices are
                1 to len(passages).

        Returns:
            Dictionary with:
                - valid: bool - True if all citations are within valid range
                - citations_found: List[int] - All unique citation numbers found
                - invalid_citations: List[int] - Citations outside valid range
                - reason: str - Human-readable explanation
        """
        logger.debug(f"Verifying citations in answer of length {len(answer)}")

        # Extract all citation numbers from the answer
        matches = self.CITATION_PATTERN.findall(answer)
        citations_found = sorted(set(int(m) for m in matches))

        logger.debug(
            f"Found {len(citations_found)} unique citations: {citations_found}"
        )

        # Determine valid range: 1 to len(passages) inclusive
        max_valid_index = len(passages)

        # Find invalid citations (out of range or <= 0)
        invalid_citations = [
            c for c in citations_found if c <= 0 or c > max_valid_index
        ]

        # Determine validity
        valid = len(invalid_citations) == 0

        # Build human-readable reason
        if not citations_found:
            reason = "No citations found in answer."
        elif valid:
            reason = f"All {len(citations_found)} citation(s) are valid."
        else:
            reason = f"Found invalid citations: {invalid_citations}"

        result = {
            "valid": valid,
            "citations_found": citations_found,
            "invalid_citations": invalid_citations,
            "reason": reason,
        }

        logger.debug(f"Citation verification result: valid={valid}, reason='{reason}'")

        return result
