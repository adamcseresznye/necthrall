import sys
import pathlib
import traceback
import asyncio

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI
from utils.embedding_manager import EmbeddingManager


# Create simple Test model (duck-typed) to avoid inheriting SentenceTransformer
class SimpleEmbeddingModel:
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


app = FastAPI()
app.state = type("S", (), {})()
app.state.embedding_model = SimpleEmbeddingModel()

emb_mgr = EmbeddingManager(app)

chunks = [
    {
        "content": "This is a test passage about blood pressure and hypertension",
        "section": "results",
        "paper_id": "p1",
    }
]

try:
    res = asyncio.run(emb_mgr.process_chunks_async(chunks))
    print("Result length:", len(res))
    if res:
        print("First embedding shape:", res[0]["embedding"].shape)
except Exception as e:
    print("Exception during embedding processing:")
    traceback.print_exc()
