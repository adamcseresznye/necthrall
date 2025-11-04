import requests
import time
import json
import os
from loguru import logger
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use loguru for structured logging


@dataclass
class Paper:
    """Represents a scientific paper from OpenAlex."""

    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    doi: Optional[str]
    type: str
    abstract: Optional[str]
    pdf_url: Optional[str]
    citation_count: int = 0


import os
import requests
import time
import json
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use loguru for structured logging


@dataclass
class Paper:
    """Represents a scientific paper from OpenAlex."""

    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    doi: Optional[str]
    type: str
    abstract: Optional[str]
    pdf_url: Optional[str]
    citation_count: int = 0


def _reconstruct_abstract(
    inverted_index: Optional[Dict[str, List[int]]],
) -> Optional[str]:
    """
    Reconstructs the abstract from OpenAlex's inverted index format.

    Args:
        inverted_index: A dictionary where keys are words and values are lists of positions.

    Returns:
        The reconstructed abstract as a string, or None if the index is None.
    """
    if inverted_index is None:
        return None
    if not inverted_index:
        return ""

    words_with_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words_with_positions.append((pos, word))

    # Sort by position
    words_with_positions.sort()

    # Join the words
    return " ".join(word for pos, word in words_with_positions)


class OpenAlexClient:
    """A client for interacting with the OpenAlex API."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, mailto: Optional[str] = None):
        """
        Initializes the OpenAlexClient.

        Args:
            mailto: A polite email address to include in the User-Agent.
                   If None, reads from OPENALEX_EMAIL environment variable.
        """
        if mailto is None:
            mailto = os.getenv("OPENALEX_EMAIL", "gemini-cli@google.com")
        self.mailto = mailto
        self.headers = {"User-Agent": f"python-requests/2.28.1 (mailto:{mailto})"}

    def search_papers(
        self,
        query: str,
        max_results: int = 20,
        retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> List[Paper]:
        """
        Generic search helper that returns a list of Paper objects.
        This method keeps behavior compatible with previous tests that call
        `search_papers` directly.
        """
        params = {
            "search": query,
            "sort": "publication_date:desc",
            "per_page": max_results,
            "select": "id,title,authorships,publication_year,primary_location,cited_by_count,best_oa_location,abstract_inverted_index,doi,type",
            "mailto": self.mailto,
        }

        for attempt in range(retries):
            try:
                time.sleep(0.1)
                resp = requests.get(
                    self.BASE_URL, headers=self.headers, params=params, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                papers = []
                for item in data.get("results", []):
                    best_oa_location = item.get("best_oa_location")
                    pdf_url = (
                        best_oa_location.get("pdf_url") if best_oa_location else None
                    )
                    authors = [
                        (author.get("author") or {}).get("display_name")
                        for author in (item.get("authorships") or [])
                        if author and author.get("author")
                    ]
                    journal_location = item.get("primary_location")
                    journal = (
                        journal_location.get("source", {}).get("display_name")
                        if journal_location and journal_location.get("source")
                        else None
                    )

                    paper = Paper(
                        paper_id=item.get("id"),
                        title=item.get("title"),
                        authors=[a for a in authors if a],
                        year=item.get("publication_year"),
                        journal=journal,
                        citation_count=item.get("cited_by_count", 0),
                        doi=(
                            item.get("doi", "").replace("https://doi.org/", "")
                            if item.get("doi")
                            else None
                        ),
                        type=item.get("type", "article"),
                        abstract=_reconstruct_abstract(
                            item.get("abstract_inverted_index")
                        ),
                        pdf_url=pdf_url,
                    )
                    papers.append(paper)

                return papers
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"OpenAlex generic search failed (attempt {attempt+1}): {e}"
                )
                time.sleep(backoff_factor * (2**attempt))
                continue

        return []

    def search_papers_with_type(
        self,
        query: str,
        paper_type: str,
        max_results: int = 20,
        retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> List[Paper]:
        """
        Searches for open-access scientific papers on OpenAlex with type filtering.

        This method will attempt progressively relaxed filters to maximize the
        chance of finding OA papers with accessible PDFs:
          1. type:<paper_type>,has_fulltext:true,is_oa:true
          2. type:<paper_type>,is_oa:true
          3. type:<paper_type>

        Returns a list of Paper dataclasses that include a pdf_url (if available).
        """
        if paper_type not in ["review", "article"]:
            raise ValueError("paper_type must be either 'review' or 'article'.")

        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100.")

        sort_order = (
            "cited_by_count:desc" if paper_type == "review" else "publication_date:desc"
        )

        filter_candidates = [
            f"type:{paper_type},has_fulltext:true,is_oa:true",
            f"type:{paper_type},is_oa:true",
            f"type:{paper_type}",
        ]

        for filter_str in filter_candidates:
            params = {
                "search": query,
                "filter": filter_str,
                "sort": sort_order,
                "per_page": max_results,
                "select": "id,title,authorships,publication_year,primary_location,cited_by_count,best_oa_location,abstract_inverted_index,doi,type",
                "mailto": self.mailto,
            }

            for attempt in range(retries):
                try:
                    time.sleep(0.1)
                    resp = requests.get(
                        self.BASE_URL, headers=self.headers, params=params, timeout=10
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    papers: List[Paper] = []
                    for item in data.get("results", []):
                        best_oa_location = item.get("best_oa_location")
                        pdf_url = (
                            best_oa_location.get("pdf_url")
                            if best_oa_location
                            else None
                        )
                        # only include papers with a pdf_url
                        if not pdf_url:
                            continue

                        authors = [
                            (author.get("author") or {}).get("display_name")
                            for author in (item.get("authorships") or [])
                            if author and author.get("author")
                        ]
                        journal_location = item.get("primary_location")
                        journal = (
                            journal_location.get("source", {}).get("display_name")
                            if journal_location and journal_location.get("source")
                            else None
                        )

                        paper = Paper(
                            paper_id=item.get("id"),
                            title=item.get("title"),
                            authors=[a for a in authors if a],
                            year=item.get("publication_year"),
                            journal=journal,
                            citation_count=item.get("cited_by_count", 0),
                            doi=(
                                item.get("doi", "").replace("https://doi.org/", "")
                                if item.get("doi")
                                else None
                            ),
                            type=item.get("type", paper_type),
                            abstract=_reconstruct_abstract(
                                item.get("abstract_inverted_index")
                            ),
                            pdf_url=pdf_url,
                        )
                        papers.append(paper)

                    if papers:
                        logger.info(
                            json.dumps(
                                {
                                    "query": query,
                                    "paper_type": paper_type,
                                    "result_count": len(papers),
                                }
                            )
                        )
                        logger.info(
                            f"OpenAlex: found {len(papers)} papers using filter '{filter_str}'"
                        )
                        return papers

                    # if no papers, break retry and try next filter
                    break

                except requests.exceptions.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 429:
                        sleep_time = backoff_factor * (2**attempt)
                        logger.warning(
                            f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds."
                        )
                        time.sleep(sleep_time)
                        continue
                    elif status and status >= 500:
                        logger.error(
                            f"OpenAlex server error: {e}. Attempt {attempt + 1} of {retries}."
                        )
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"HTTP error occurred: {e}")
                        return []
                except requests.exceptions.RequestException as e:
                    logger.error(f"An error occurred while searching OpenAlex: {e}")
                    return []
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response: {e}")
                    return []

        # no papers found across filters
        logger.info(
            json.dumps({"query": query, "paper_type": paper_type, "result_count": 0})
        )
        return []
