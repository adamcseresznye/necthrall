import pytest
import numpy as np
from unittest.mock import Mock
from datetime import datetime
from uuid import uuid4

from models.state import State, Paper, Passage, Score, PDFContent


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
    from models.state import ErrorReport

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


# Integration Testing Fixtures


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI app with cached embedding model for ProcessingAgent."""
    from unittest.mock import MagicMock

    app = Mock()
    app.state = Mock()

    # Create a mock that properly inherits SentenceTransformer behavior
    class MockSentenceTransformer(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.encode = Mock(
                side_effect=lambda texts, **kwargs: np.array(
                    [[0.1 + i * 0.01] * 384 for i, _ in enumerate(texts)],
                    dtype=np.float32,
                )
            )

        def __class__(self):
            # Return the actual SentenceTransformer class for isinstance checks
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer

    # Make isinstance and type checks work
    mock_model = MockSentenceTransformer()

    # Patch isinstance to treat our mock as a real SentenceTransformer
    original_isinstance = isinstance

    def patched_isinstance(obj, cls):
        if (
            obj is mock_model
            and hasattr(cls, "__name__")
            and cls.__name__ == "SentenceTransformer"
        ):
            return True
        return original_isinstance(obj, cls)

    # Patch globally for this test session
    import builtins

    builtins.isinstance = patched_isinstance

    app.state.embedding_model = mock_model
    return app


@pytest.fixture
def realistic_scientific_papers():
    """Realistic mock scientific papers based on intermittent fasting research."""
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
    """Realistic PDF content with IMRaD structure matching the papers."""
    content1 = """1. Introduction

Cardiovascular disease remains a leading cause of mortality worldwide, accounting for approximately 17.9 million deaths annually. Intermittent fasting, characterized by alternating periods of eating and fasting, has emerged as a potential intervention for metabolic health and cardiovascular risk reduction. This comprehensive systematic review and meta-analysis examines the evidence for cardiovascular effects of intermittent fasting in human populations, drawing from randomized controlled trials and observational studies conducted over the past decade.

The rising prevalence of obesity and related cardiovascular complications necessitates novel therapeutic approaches beyond traditional pharmacological interventions. Intermittent fasting represents one such strategy that combines dietary modification with chronobiological alignment, potentially leveraging evolutionary adaptations to food scarcity.

2. Methods

We conducted a systematic review of randomized controlled trials published between 2000-2023 investigating intermittent fasting effects on cardiovascular outcomes. Multiple databases were comprehensively searched: PubMed, Cochrane Central Register of Controlled Trials, Web of Science, Scopus, and ClinicalTrials.gov. Search terms included "intermittent fasting", "time-restricted eating", "cardiovascular disease", "blood pressure", "hypertension", "dyslipidemia", "glucose metabolism", and related medical subject headings.

3. Results

Twenty-four randomized trials involving 2,847 participants were included. Intermittent fasting significantly reduced systolic blood pressure by -4.8 mmHg and diastolic blood pressure by -3.1 mmHg. Lipid profile improvements included total cholesterol decrease of -0.31 mmol/L, LDL cholesterol reduction of -0.25 mmol/L, and triglycerides reduction of -0.23 mmol/L.

Notably, cardiovascular risk factors like elevated blood pressure and dyslipidemia showed greater improvements with longer fasting durations (>16 hours/day) compared to shorter protocols. Anti-inflammatory effects were observed with decreased C-reactive protein and interleukin-6 levels.

4. Discussion

The cardiovascular benefits of intermittent fasting appear mediated through multiple physiological mechanisms including weight loss, enhanced insulin sensitivity, autonomic nervous system modulation, and anti-inflammatory effects. Subgroup analyses revealed greater benefits in participants with higher baseline cardiovascular risk.

Study limitations include heterogeneous fasting protocols, variable intervention durations, and inconsistent outcome reporting. Publication bias toward positive results may overestimate effect sizes.

5. Conclusion

Intermittent fasting represents a promising dietary approach for cardiovascular risk reduction with evidence from multiple randomized trials. Implementation should consider patient preferences and tolerability."""

    content2 = """INTRODUCTION

Time-restricted eating (TRE), a form of intermittent fasting, has shown promise in improving metabolic parameters including blood pressure control. This randomized controlled trial evaluated the effects of different TRE protocols on 24-hour ambulatory blood pressure monitoring in hypertensive adults.

METHODS AND MATERIALS

Eighty hypertensive participants (SBP 135-160 mmHg) were randomized to one of three groups: 16-hour TRE (eating window 12:00-20:00), 14-hour TRE (eating window 10:00-24:00), or usual care. Blood pressure was assessed using 24-hour ambulatory monitoring at baseline, 8 weeks, and 16 weeks. The TRE groups received standardized dietary counseling combined with timing restrictions.

RESULTS

At 16 weeks, the 16-hour TRE group demonstrated significant reductions in 24-hour systolic blood pressure (-8.2 ± 2.3 mmHg) and diastolic blood pressure (-5.1 ± 1.8 mmHg) compared to usual care. The 14-hour TRE group showed moderate improvements (-4.8 ± 2.8 mmHg SBP, -3.2 ± 1.9 mmHg DBP).

Cardiovascular risk markers including pulse wave velocity and inflammatory markers also improved significantly in the longer fasting duration group. No significant changes were observed in lipid profiles during the study period.

QUALITY OF LIFE AND SAFETY

Both TRE protocols were well-tolerated with high adherence rates (87% completion). Quality of life measures remained stable, and no serious adverse cardiovascular events occurred. Common side effects were mild hunger and transient fatigue.

DISCUSSION

These findings support the use of longer-duration TRE protocols for blood pressure management in hypertensive adults. The circadian alignment of eating patterns may contribute to improved autonomic function and reduced cardiovascular risk. Further studies are needed to determine optimal fasting duration and long-term cardiovascular outcomes.

CONCLUSION

16-hour time-restricted eating significantly improves 24-hour ambulatory blood pressure in hypertensive adults, suggesting potential as an adjunct therapy for cardiovascular risk reduction."""

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
    """Predefined test queries with expected relevance patterns."""
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
                "cardiovascular",
                "ambulatory",
                "risk",
            ],
            "min_relevant_passages": 2,
            "description": "Should retrieve from TRE trial and review",
        },
    ]


@pytest.fixture
def processing_integration_state(realistic_scientific_papers, realistic_pdf_contents):
    """Complete State fixture for ProcessingAgent integration testing."""
    state = State(
        original_query="cardiovascular effects of intermittent fasting",
        optimized_query="cardiovascular effects of intermittent fasting",
        filtered_papers=realistic_scientific_papers,
        pdf_contents=realistic_pdf_contents,
    )
    return state
