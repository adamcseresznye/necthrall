import pytest
from datetime import datetime
from uuid import uuid4

from models.state import State, Paper, Passage, Score


@pytest.fixture
def initial_state() -> State:
    """Provides a basic State object with a query."""
    return State(original_query="What is the role of autophagy in neurodegeneration?")


@pytest.fixture
def search_state(initial_state: State) -> State:
    """Provides a State object after the search step."""
    paper1 = Paper(
        paper_id="openalex:p1",
        title="Paper 1 on Autophagy",
        authors=["Author A", "Author B"],
        year=2023,
        journal="Nature",
        citation_count=150,
        doi="10.1038/nature12345",
        pdf_url="http://example.com/p1.pdf",
        type="article",
    )
    paper2 = Paper(
        paper_id="openalex:p2",
        title="Paper 2 on Neurodegeneration",
        authors=["Author C"],
        year=2023,
        journal="Science",
        citation_count=89,
        doi="10.1126/science67890",
        pdf_url="http://example.com/p2.pdf",
        type="review",
    )
    initial_state.papers_metadata = [paper1, paper2]
    initial_state.search_quality = {
        "passed": True,
        "reason": "Found 2 papers with avg_relevance 0.85",
        "paper_count": 2,
        "avg_relevance": 0.85,
    }
    return initial_state


@pytest.fixture
def acquisition_state(search_state: State) -> State:
    """Provides a State object after the acquisition step."""
    # Add some PDF content to simulate acquisition
    from models.state import PDFContent, ErrorReport

    search_state.pdf_contents = [
        PDFContent(
            paper_id="openalex:p1",
            raw_text="Autophagy is a key process for cellular homeostasis.",
            page_count=10,
            char_count=500,
            extraction_time=1.2,
        )
    ]
    search_state.download_failures = [
        ErrorReport(
            paper_id="openalex:p2",
            url="http://example.com/p2.pdf",
            error_type="HTTPError",
            message="404 Not Found",
            timestamp=1234567890.0,
            recoverable=True,
        )
    ]
    return search_state


@pytest.fixture
def processing_state(acquisition_state: State) -> State:
    """Provides a State object after the processing step."""
    passage1 = Passage(
        passage_id="ps1",
        paper_id="openalex:p1",
        text="Autophagy is a key process for cellular homeostasis.",
        page_number=1,
        char_start=0,
        char_end=50,
    )
    passage2 = Passage(
        passage_id="ps2",
        paper_id="openalex:p1",
        text="Dysfunctional autophagy is linked to many neurodegenerative diseases.",
        page_number=2,
        char_start=51,
        char_end=120,
    )
    acquisition_state.passages = [passage1, passage2]
    return acquisition_state


@pytest.fixture
def analysis_state(processing_state: State) -> State:
    """Provides a State object after the analysis step."""
    score1 = Score(
        score_type="relevance",
        value=0.95,
        justification="Directly addresses the query about autophagy and neurodegeneration.",
    )
    score2 = Score(
        score_type="novelty",
        value=0.80,
        justification="Presents a new perspective on cellular mechanisms.",
    )

    # Store scores in the scores dict with appropriate keys
    processing_state.scores = {"relevance": [score1], "novelty": [score2]}
    return processing_state
