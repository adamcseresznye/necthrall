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
        sort_by: Optional[str] = None,
    ) -> List[Paper]:
        """
        Generic search helper that returns a list of Paper objects.
        This method keeps behavior compatible with previous tests that call
        `search_papers` directly.
        """
        params = {
            "search": query,
            "sort": sort_by or "publication_date:desc",
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
                try:
                    data = resp.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response: {e}")
                    return []

                papers = []
                for item in data.get("results", []):
                    best_oa_location = item.get("best_oa_location")
                    pdf_url = (
                        best_oa_location.get("pdf_url") if best_oa_location else None
                    )
                    # only include papers with a pdf_url (behavior expected by tests)
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
                        else "Unknown"
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
                    # server error: small fixed backoff as tests expect
                    logger.error(
                        f"OpenAlex server error: {e}. Attempt {attempt + 1} of {retries}."
                    )
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"HTTP error occurred: {e}")
                    return []
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
        paginate: bool = False,
        per_page: int = 50,
        require_pdf: bool = False,
        sort_by: Optional[str] = None,
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

        # allow larger max_results when paginating, otherwise keep a sensible cap
        if paginate:
            if max_results < 1 or max_results > 1000:
                raise ValueError(
                    "max_results must be between 1 and 1000 when paginate=True."
                )
        else:
            if not 1 <= max_results <= 100:
                raise ValueError(
                    "max_results must be between 1 and 100 when paginate=False."
                )

        if not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100.")

        # Default sorting: for reviews prefer citations, for articles prefer recent.
        # Allow caller override via sort_by parameter.
        sort_order = sort_by or (
            "cited_by_count:desc" if paper_type == "review" else "publication_date:desc"
        )

        # Build filter candidates. By default we do NOT require PDFs; callers can set
        # `require_pdf=True` to prefer items that have a PDF URL.
        filter_candidates = []
        if require_pdf:
            filter_candidates.append(f"type:{paper_type},has_fulltext:true,is_oa:true")
        filter_candidates.append(f"type:{paper_type},is_oa:true")
        filter_candidates.append(f"type:{paper_type}")

        select_fields = (
            "id,title,authorships,publication_year,primary_location,cited_by_count,"
            "best_oa_location,abstract_inverted_index,doi,type"
        )

        # If pagination is requested, use per_page and loop pages until we reach max_results
        if paginate:
            papers: List[Paper] = []
            page = 1
            # try filters in order until any returns results
            for filter_str in filter_candidates:
                page = 1
                papers = []
                while len(papers) < max_results:
                    params = {
                        "search": query,
                        "filter": filter_str,
                        "sort": sort_order,
                        "per_page": per_page,
                        "page": page,
                        "select": select_fields,
                        "mailto": self.mailto,
                    }

                    for attempt in range(retries):
                        try:
                            time.sleep(0.1)
                            resp = requests.get(
                                self.BASE_URL,
                                headers=self.headers,
                                params=params,
                                timeout=10,
                            )
                            resp.raise_for_status()
                            data = resp.json()

                            results = data.get("results", [])
                            for item in results:
                                best_oa_location = item.get("best_oa_location")
                                pdf_url = (
                                    best_oa_location.get("pdf_url")
                                    if best_oa_location
                                    else None
                                )
                                # if caller requested only papers with pdfs, skip those without
                                if require_pdf and not pdf_url:
                                    continue

                                authors = [
                                    (author.get("author") or {}).get("display_name")
                                    for author in (item.get("authorships") or [])
                                    if author and author.get("author")
                                ]
                                journal_location = item.get("primary_location")
                                journal = (
                                    journal_location.get("source", {}).get(
                                        "display_name"
                                    )
                                    if journal_location
                                    and journal_location.get("source")
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
                                        item.get("doi", "").replace(
                                            "https://doi.org/", ""
                                        )
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
                                if len(papers) >= max_results:
                                    break

                            meta = data.get("meta", {})
                            total_available = meta.get("count")
                            # stop if no more results available
                            if not results or (
                                total_available is not None
                                and page * per_page >= total_available
                            ):
                                break

                            page += 1
                            time.sleep(0.1)
                            break  # successful attempt, break retry loop

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
                                papers = []
                                break
                        except requests.exceptions.RequestException as e:
                            logger.error(
                                f"An error occurred while searching OpenAlex: {e}"
                            )
                            papers = []
                            break
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON response: {e}")
                            papers = []
                            break

                    # if we collected any papers for this filter, return them (or continue to next filter)
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
                            f"OpenAlex: found {len(papers)} papers using filter '{filter_str}' (paginated)"
                        )
                        return papers[:max_results]

                # try next filter candidate

            # no papers found across filters
            logger.info(
                json.dumps(
                    {"query": query, "paper_type": paper_type, "result_count": 0}
                )
            )
            return []

        # Non-paginated (single request) behaviour: try filter candidates
        for filter_str in filter_candidates:
            params = {
                "search": query,
                "filter": filter_str,
                "sort": sort_order,
                "per_page": max_results,
                "select": select_fields,
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
                        if require_pdf and not pdf_url:
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

        logger.info(
            json.dumps({"query": query, "paper_type": paper_type, "result_count": 0})
        )
        return []
