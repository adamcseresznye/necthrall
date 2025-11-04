import sys
import pathlib
from fastapi import FastAPI
import numpy as np

# Ensure project root is on sys.path for local imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.processing_agent import ProcessingAgent
from models.state import Paper, PDFContent, State

# Lightweight TestSentenceTransformer (same as in fixtures)
try:
    from sentence_transformers import SentenceTransformer

    class TestSentenceTransformer(SentenceTransformer):
        def __init__(self):
            pass

        def encode(
            self,
            texts,
            batch_size=None,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
            **kwargs
        ):
            if isinstance(texts, str):
                texts = [texts]
            return np.array(
                [[0.1 + i * 0.01] * 384 for i, _ in enumerate(texts)], dtype=np.float32
            )

    mock_model = TestSentenceTransformer()
except Exception:

    class TestSentenceTransformer:
        def encode(
            self,
            texts,
            batch_size=None,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
            **kwargs
        ):
            if isinstance(texts, str):
                texts = [texts]
            return np.array(
                [[0.1 + i * 0.01] * 384 for i, _ in enumerate(texts)], dtype=np.float32
            )

    mock_model = TestSentenceTransformer()

# Build app
app = FastAPI()
app.state = type("S", (), {})()
app.state.embedding_model = mock_model

# Create fake papers and pdf contents similar to fixtures
paper1 = Paper(
    paper_id="openalex:fasting-cardio-1",
    title="Cardiovascular Effects of Intermittent Fasting",
    authors=["A"],
    year=2023,
    journal="Journal",
    citation_count=100,
    doi=None,
    pdf_url=None,
    type="article",
)
paper2 = Paper(
    paper_id="openalex:fasting-cardio-2",
    title="Time-Restricted Eating and Blood Pressure Control",
    authors=["B"],
    year=2023,
    journal="Journal",
    citation_count=50,
    doi=None,
    pdf_url=None,
    type="article",
)

content1 = "Cardiovascular disease remains a leading cause of mortality worldwide. Studies report changes in blood pressure, systolic and diastolic measures, and links to hypertension as key risk factors."
content2 = "Time-restricted eating (TRE) has shown promise for reducing blood pressure and improving cardiovascular biomarkers. Clinical trials report reductions in systolic and diastolic blood pressure."

pdf1 = PDFContent(
    paper_id=paper1.paper_id,
    raw_text=content1,
    page_count=10,
    char_count=len(content1),
    extraction_time=0.1,
)
pdf2 = PDFContent(
    paper_id=paper2.paper_id,
    raw_text=content2,
    page_count=8,
    char_count=len(content2),
    extraction_time=0.1,
)

state = State(
    original_query="cardiovascular risks of fasting",
    optimized_query="cardiovascular risks of fasting",
    filtered_papers=[paper1, paper2],
    pdf_contents=[pdf1, pdf2],
)

agent = ProcessingAgent(app)
result_state = agent(state)

print("Top passages:")
for p in result_state.top_passages[:10]:
    print("-", p.get("content")[:200])

print("\nProcessing stats:")
print(result_state.processing_stats)
