import requests
import time
import logging
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __init__(self, mailto: str = "gemini-cli@google.com"):
        """
        Initializes the OpenAlexClient.

        Args:
            mailto: A polite email address to include in the User-Agent.
        """
        self.headers = {"User-Agent": f"python-requests/2.28.1 (mailto:{mailto})"}

    def search_papers(
        self,
        query: str,
        max_results: int = 20,
        retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> List[Paper]:
        """
        Searches for open-access scientific papers on OpenAlex.

        Args:
            query: The scientific query to search for.
            max_results: The maximum number of papers to return (1-25).
            retries: The number of times to retry the request.
            backoff_factor: The factor to use for exponential backoff.

        Returns:
            A list of Paper objects.
        """
        if not 1 <= max_results <= 25:
            raise ValueError("max_results must be between 1 and 25.")

        params = {
            "search": query,
            "filter": "has_fulltext:true,is_oa:true",
            "sort": "relevance_score:desc",  # Prioritize highly cited papers
            "per_page": max_results,
            "select": "id,title,authorships,publication_year,primary_location,cited_by_count,best_oa_location,abstract_inverted_index,doi,type",
        }

        start_time = time.time()
        for attempt in range(retries):
            try:
                time.sleep(0.1)  # Respect rate limit
                response = requests.get(
                    self.BASE_URL, headers=self.headers, params=params, timeout=10
                )
                response.raise_for_status()

                data = response.json()

                papers = []
                for item in data.get("results", []):
                    best_oa_location = item.get("best_oa_location")
                    if not best_oa_location or not best_oa_location.get("pdf_url"):
                        continue

                    authors = [
                        author["author"]["display_name"]
                        for author in item.get("authorships", [])
                        if author.get("author")
                    ]
                    journal_location = item.get("primary_location")
                    journal = (
                        journal_location.get("source", {}).get("display_name")
                        if journal_location and journal_location.get("source")
                        else "Unknown"
                    )

                    paper = Paper(
                        paper_id=item.get("id"),
                        title=item.get("title"),
                        authors=authors,
                        year=item.get("publication_year"),
                        journal=journal,
                        citation_count=item.get("cited_by_count", 0),
                        doi=(
                            item.get("doi", "").replace("https://doi.org/", "")
                            if item.get("doi")
                            else None
                        ),
                        pdf_url=best_oa_location.get("pdf_url"),
                        abstract=_reconstruct_abstract(
                            item.get("abstract_inverted_index")
                        ),
                        type=item.get(
                            "type", "article"
                        ),  # Use actual type from OpenAlex
                    )
                    papers.append(paper)

                duration = time.time() - start_time
                log_data = {
                    "query": query,
                    "result_count": len(papers),
                    "duration_ms": round(duration * 1000),
                }
                logger.info(json.dumps(log_data))
                return papers

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    sleep_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds."
                    )
                    time.sleep(sleep_time)
                    continue
                elif e.response.status_code >= 500:
                    logger.error(
                        f"OpenAlex server error: {e}. Attempt {attempt + 1} of {retries}."
                    )
                    time.sleep(1)  # Wait a second before retrying on server error
                    continue
                else:
                    logger.error(f"HTTP error occurred: {e}")
                    return []
            except requests.exceptions.Timeout as e:
                logger.error(
                    f"Request timed out: {e}. Attempt {attempt + 1} of {retries}."
                )
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return []
            except requests.exceptions.RequestException as e:
                logger.error(f"An error occurred while searching OpenAlex: {e}")
                return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response: {e}")
                return []
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse OpenAlex response: {e}")
                return []

        logger.error("All retries failed.")
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

        Args:
            query: The scientific query to search for.
            paper_type: The type of paper to search for ("review" or "article").
            max_results: The maximum number of papers to return (1-100).
            retries: The number of times to retry the request.
            backoff_factor: The factor to use for exponential backoff.

        Returns:
            A list of Paper objects.
        """
        if paper_type not in ["review", "article"]:
            raise ValueError("paper_type must be either 'review' or 'article'.")

        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100.")

        # Set sort order based on paper type
        sort_order = (
            "cited_by_count:desc" if paper_type == "review" else "publication_date:desc"
        )

        params = {
            "search": query,
            "filter": f"type:{paper_type},has_fulltext:true,is_oa:true",
            "sort": sort_order,
            "per_page": max_results,
            "select": "id,title,authorships,publication_year,primary_location,cited_by_count,best_oa_location,abstract_inverted_index,doi,type",
            "mailto": "your-email@example.com",  # Polite pool for higher rate limits
        }

        start_time = time.time()
        for attempt in range(retries):
            try:
                time.sleep(0.1)  # Respect rate limit
                response = requests.get(
                    self.BASE_URL, headers=self.headers, params=params, timeout=10
                )
                response.raise_for_status()

                data = response.json()

                papers = []
                for item in data.get("results", []):
                    best_oa_location = item.get("best_oa_location")
                    if not best_oa_location or not best_oa_location.get("pdf_url"):
                        continue

                    authors = [
                        author["author"]["display_name"]
                        for author in item.get("authorships", [])
                        if author.get("author")
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
                        authors=authors,
                        year=item.get("publication_year"),
                        journal=journal,
                        citation_count=item.get("cited_by_count", 0),
                        doi=(
                            item.get("doi", "").replace("https://doi.org/", "")
                            if item.get("doi")
                            else None
                        ),
                        pdf_url=best_oa_location.get("pdf_url"),
                        abstract=_reconstruct_abstract(
                            item.get("abstract_inverted_index")
                        ),
                        type=item.get("type", paper_type),
                    )
                    papers.append(paper)

                duration = time.time() - start_time
                log_data = {
                    "query": query,
                    "paper_type": paper_type,
                    "result_count": len(papers),
                    "duration_ms": round(duration * 1000),
                }
                logger.info(json.dumps(log_data))
                return papers

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    sleep_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds."
                    )
                    time.sleep(sleep_time)
                    continue
                elif e.response.status_code >= 500:
                    logger.error(
                        f"OpenAlex server error: {e}. Attempt {attempt + 1} of {retries}."
                    )
                    time.sleep(1)  # Wait a second before retrying on server error
                    continue
                else:
                    logger.error(f"HTTP error occurred: {e}")
                    return []
            except requests.exceptions.Timeout as e:
                logger.error(
                    f"Request timed out: {e}. Attempt {attempt + 1} of {retries}."
                )
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return []
            except requests.exceptions.RequestException as e:
                logger.error(f"An error occurred while searching OpenAlex: {e}")
                return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response: {e}")
                return []
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse OpenAlex response: {e}")
                return []

        logger.error("All retries failed.")
        return []
