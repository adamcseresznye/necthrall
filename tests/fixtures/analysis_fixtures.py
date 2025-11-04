import json
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

from models.state import (
    Paper,
    Passage,
    State,
    ProcessingConfig,
    PDFContent,
    ErrorReport,
    Score,
)


@pytest.fixture
def high_cred_paper_dict():
    return {
        "paper_id": "nature_2023",
        "citation_count": 250,
        "year": 2023,
        "journal": "Nature",
    }


@pytest.fixture
def medium_cred_paper_dict():
    return {
        "paper_id": "mid_2020",
        "citation_count": 45,
        "year": 2020,
        "journal": "Journal of Testing",
    }


@pytest.fixture
def low_cred_paper_dict():
    return {
        "paper_id": "preprint_2015",
        "citation_count": 3,
        "year": 2015,
        "journal": "arXiv",
    }


@pytest.fixture
def high_cred_paper():
    return Paper(
        paper_id="nature_2023",
        title="Great discovery",
        authors=["A"],
        year=2023,
        journal="Nature",
        citation_count=250,
        pdf_url=None,
        type="article",
    )


@pytest.fixture
def medium_cred_paper():
    return Paper(
        paper_id="mid_2020",
        title="Mid work",
        authors=["B"],
        year=2020,
        journal="Journal of Testing",
        citation_count=45,
        pdf_url=None,
        type="article",
    )


@pytest.fixture
def low_cred_paper():
    return Paper(
        paper_id="preprint_2015",
        title="Early draft",
        authors=["C"],
        year=2015,
        journal="arXiv",
        citation_count=3,
        pdf_url=None,
        type="article",
    )


@pytest.fixture
def contradictory_passages():
    # Two passages with opposing claims
    p1 = Passage(
        content="We found that drug X increases survival by 50%.",
        section="results",
        paper_id="p1",
        retrieval_score=0.95,
    )
    p2 = Passage(
        content="Our analysis shows drug X has no effect on survival.",
        section="results",
        paper_id="p2",
        retrieval_score=0.93,
    )
    return [p1, p2]


@pytest.fixture
def valid_llm_response_json():
    # Minimal valid DetectedContradiction list as JSON string
    data = [
        {
            "topic": "effect_of_drug_x",
            "claim_1": {"paper_id": "p1", "text": "drug X increases survival"},
            "claim_2": {"paper_id": "p2", "text": "drug X has no effect"},
            "severity": "major",
        }
    ]
    return json.dumps(data)


@pytest.fixture
def malformed_llm_response():
    return "{this is: not valid json]"


@pytest.fixture
def langgraph_state_factory():
    def _factory(
        filtered_papers=None, relevant_passages=None, query="What about drug X?"
    ):
        state = State(original_query=query)
        if filtered_papers:
            state.filtered_papers = filtered_papers
        if relevant_passages:
            state.relevant_passages = relevant_passages
        # Minimal execution_times container
        state.execution_times = {}
        state.config = ProcessingConfig()
        return state

    return _factory


# --- Fixtures migrated from legacy tests/fixtures.py ---


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
        content="Autophagy is a key process for cellular homeostasis.",
        section="results",
        paper_id="openalex:p1",
        retrieval_score=0.95,
    )
    passage2 = Passage(
        content="Dysfunctional autophagy is linked to many neurodegenerative diseases.",
        section="results",
        paper_id="openalex:p1",
        retrieval_score=0.92,
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

    processing_state.scores = {"relevance": [score1], "novelty": [score2]}
    return processing_state


@pytest.fixture
def mock_fastapi_app(request):
    """Mock FastAPI app with cached embedding model for ProcessingAgent."""
    app = Mock()
    app.state = Mock()

    # Provide a lightweight test implementation that subclasses the real
    # SentenceTransformer class so that isinstance checks pass in
    # EmbeddingManager._validate_model. We avoid calling the heavy
    # SentenceTransformer.__init__ to keep this fast and side-effect free.
    # Provide a simple duck-typed model (no heavy initialization). We'll
    # monkeypatch EmbeddingManager._validate_model in tests to return this
    # model so the production isinstance checks are bypassed in the test
    # environment. This keeps tests fast and deterministic.
    class SimpleEmbeddingModel:
        def encode(
            self,
            texts,
            batch_size=None,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
            **kwargs,
        ):
            if isinstance(texts, str):
                texts = [texts]
            return np.array(
                [[0.1 + i * 0.01] * 384 for i, _ in enumerate(texts)], dtype=np.float32
            )

    mock_model = SimpleEmbeddingModel()

    # Monkeypatch EmbeddingManager._validate_model to accept our simple model
    try:
        from utils.embedding_manager import EmbeddingManager
    except Exception:
        EmbeddingManager = None

    original_validate = None
    if EmbeddingManager is not None:
        original_validate = EmbeddingManager._validate_model

        def _validate_override(self):
            return mock_model

        EmbeddingManager._validate_model = _validate_override

        # Ensure the original method is restored after the test
        def _restore():
            EmbeddingManager._validate_model = original_validate

        request.addfinalizer(_restore)

    app.state.embedding_model = mock_model
    return app


@pytest.fixture
def realistic_scientific_papers():
    paper1 = Paper(
        paper_id="openalex:fasting-cardio-1",
        title="Cardiovascular Effects of Intermittent Fasting: A Systematic Review and Meta-Analysis",
        authors=["Zhang J", "Li F", "Wang Y"],
        year=2023,
        journal="Journal of the American College of Cardiology",
        citation_count=245,
        doi="10.1016/j.jacc.2023.02.045",
        pdf_url="https://example.com/fasting-cardio-1.pdf",
        type="article",
    )
    paper2 = Paper(
        paper_id="openalex:fasting-cardio-2",
        title="Time-Restricted Eating and Blood Pressure Control: Results from the TREAT Trial",
        authors=["Harris M", "Garcia S", "Thomas R"],
        year=2023,
        journal="Hypertension",
        citation_count=189,
        doi="10.1161/HYPERTENSIONAHA.122.20880",
        pdf_url="https://example.com/fasting-cardio-2.pdf",
        type="article",
    )
    return [paper1, paper2]


@pytest.fixture
def realistic_pdf_contents():
    content1 = """1. Introduction

Cardiovascular disease remains a leading cause of mortality worldwide. Studies report
changes in blood pressure, systolic and diastolic measures, and links to hypertension
as key risk factors in cardiovascular outcomes for dietary interventions."""

    content2 = """INTRODUCTION

Time-restricted eating (TRE), a form of intermittent fasting, has shown promise
for reducing blood pressure and improving cardiovascular biomarkers. Clinical
trials report modest reductions in systolic and diastolic blood pressure and
improvements in hypertension-related outcomes in some cohorts."""

    return [
        PDFContent(
            paper_id="openalex:fasting-cardio-1",
            raw_text=content1,
            page_count=12,
            char_count=len(content1),
            extraction_time=1.2,
        ),
        PDFContent(
            paper_id="openalex:fasting-cardio-2",
            raw_text=content2,
            page_count=8,
            char_count=len(content2),
            extraction_time=0.9,
        ),
    ]


@pytest.fixture
def integration_test_queries():
    return [
        {
            "query": "cardiovascular risks of fasting",
            "expected_terms_in_top5": [
                "cardiovascular",
                "blood pressure",
                "hypertension",
                "risk",
            ],
            "min_relevant_passages": 3,
            "description": "Should retrieve passages about cardiovascular effects and risks",
        },
        {
            "query": "intermittent fasting blood pressure effects",
            "expected_terms_in_top5": [
                "blood pressure",
                "systolic",
                "diastolic",
                "hypertension",
            ],
            "min_relevant_passages": 4,
            "description": "Should focus on BP effects from both studies",
        },
        {
            "query": "time restricted eating cardiovascular outcomes",
            "expected_terms_in_top5": [
                "time-restricted",
                "blood pressure",
                "cardiovascular",
                "hypertension",
            ],
            "min_relevant_passages": 2,
            "description": "Should retrieve TRE-specific cardiovascular outcomes",
        },
    ]


@pytest.fixture
def processing_integration_state(realistic_scientific_papers, realistic_pdf_contents):
    state = State(
        original_query="cardiovascular effects of intermittent fasting",
        optimized_query="cardiovascular effects of intermittent fasting",
        filtered_papers=realistic_scientific_papers,
        pdf_contents=realistic_pdf_contents,
    )
    return state


@pytest.fixture
def realistic_passages_factory():
    """
    Factory that generates realistic Passage objects mimicking OpenAlex/PDF extraction outputs.

    Usage:
        passages = realistic_passages_factory(count=5, seed=1)
    """

    def _factory(count: int = 5, seed: int = None):
        import random

        from models.state import Passage

        rnd = random.Random(seed)
        sections = ["introduction", "methods", "results", "discussion", "other"]
        passages = []
        for i in range(count):
            section = rnd.choice(sections)
            paper_id = f"openalex:sim_paper_{rnd.randint(1,100)}"
            # Create variable-length content that mimics scientific sentences
            sentences = []
            n_sent = rnd.randint(2, 6)
            for s in range(n_sent):
                # include quantitative tokens sometimes
                if rnd.random() < 0.4:
                    sentence = (
                        f"We found a {rnd.randint(1,95)}% change in outcome (p<0.05)."
                    )
                else:
                    sentence = "This study indicates a notable effect on the measured endpoint."
                sentences.append(sentence)

            content = " ".join(sentences)
            # rough token estimate (4 chars per token approx in other code)
            token_count = max(1, len(content) // 4)

            passage = Passage(
                content=content,
                section=section,
                paper_id=paper_id,
                retrieval_score=round(rnd.random(), 3),
                char_start=0,
                char_end=len(content),
            )
            # attach token_count on metadata for tests that inspect it
            passage.__dict__["token_count"] = token_count
            passages.append(passage)

        return passages

    return _factory
