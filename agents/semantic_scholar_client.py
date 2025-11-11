"""Stub Semantic Scholar client for retrieval stage.

This is intentionally minimal: a class definition and docstring so future
implementations can plug in network calls and parsing logic.
"""

from typing import Any, Dict, List


class SemanticScholarClient:
    """Client stub wrapping Semantic Scholar API interactions.

    Methods should be implemented to fetch papers for a given query and
    return structured dicts compatible with the pipeline's `State.papers`.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for `query` and return list of raw hits.

        Currently a stub that should be implemented in future work.
        """
        raise NotImplementedError("SemanticScholarClient.search is not implemented")


__all__ = ["SemanticScholarClient"]
