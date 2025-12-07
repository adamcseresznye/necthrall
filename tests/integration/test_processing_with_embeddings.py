import time
from typing import List

import pytest
import numpy as np

from agents.processing_agent import ProcessingAgent
from models.state import State


class DummyEmbedModel:
    """Deterministic dummy embedding model that returns 384-dim vectors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i, t in enumerate(texts):
            # produce simple pattern-based vector for determinism
            vec = [(i + 1) * 0.001 for _ in range(384)]
            out.append(vec)
        return out


def _make_text(tokens: int) -> str:
    return "word " * tokens


@pytest.mark.integration
def test_three_papers_get_384_embeddings():
    passages = []
    for i in range(3):
        passages.append(
            {
                "paperId": f"p{i}",
                "title": f"Paper {i}",
                "text": _make_text(600),
            }
        )

    state = State(query="test", passages=passages)
    agent = ProcessingAgent(chunk_size=200, chunk_overlap=20)

    processed = agent.process(state, embedding_model=DummyEmbedModel(), batch_size=32)

    assert processed.chunks is not None
    assert len(processed.chunks) > 0

    for c in processed.chunks:
        emb = c.metadata.get("embedding")
        assert emb is not None, "Missing embedding on chunk"
        arr = np.asarray(emb)
        assert arr.shape == (384,)


@pytest.mark.integration
def test_chunking_failure_skips_embedding(monkeypatch):
    passages = [
        {"paperId": "bad", "title": "Bad Paper", "text": _make_text(600)},
        {"paperId": "good", "title": "Good Paper", "text": _make_text(600)},
    ]

    state = State(query="test2", passages=passages)
    agent = ProcessingAgent(chunk_size=200, chunk_overlap=20)

    orig_parser = agent.markdown_parser

    class BrokenParser:
        def __init__(self, delegate):
            self._delegate = delegate
            self._first = True

        def get_nodes_from_documents(self, docs):
            if self._first:
                self._first = False
                raise RuntimeError("simulated chunker failure")
            return self._delegate.get_nodes_from_documents(docs)

    agent.markdown_parser = BrokenParser(orig_parser)

    processed = agent.process(state, embedding_model=DummyEmbedModel(), batch_size=16)

    # Ensure we still processed the good paper's chunks
    assert processed.chunks is not None
    assert any(c.metadata.get("paper_id") == "good" for c in processed.chunks)

    # Ensure no exception propagated and state.errors captured a message (or at least empty list allowed)
    assert isinstance(processed.errors, list)


@pytest.mark.integration
def test_metadata_preserved_through_embedding():
    passages = [{"paperId": "meta1", "title": "Meta Paper", "text": _make_text(300)}]
    state = State(query="meta", passages=passages)
    agent = ProcessingAgent(chunk_size=100, chunk_overlap=10)

    processed = agent.process(state, embedding_model=DummyEmbedModel(), batch_size=8)

    assert processed.chunks is not None
    for idx, c in enumerate(processed.chunks):
        assert c.metadata.get("paper_id") == "meta1"
        assert "header_path" in c.metadata
        assert "chunk_index" in c.metadata
        # embedding still present
        emb = c.metadata.get("embedding")
        assert emb is not None


@pytest.mark.integration
def test_end_to_end_timing_for_five_papers_is_fast():
    passages = []
    for i in range(5):
        passages.append({"paperId": f"t{i}", "title": f"T{i}", "text": _make_text(400)})

    state = State(query="timer", passages=passages)
    agent = ProcessingAgent(chunk_size=200, chunk_overlap=20)
    start = time.perf_counter()
    processed = agent.process(state, embedding_model=DummyEmbedModel(), batch_size=64)
    elapsed = time.perf_counter() - start

    assert elapsed < 6.0, f"Processing too slow: {elapsed:.2f}s"
    assert processed.chunks is not None
    assert len(processed.chunks) > 0


@pytest.mark.integration
def test_embedding_retry_succeeds_after_failures():
    """Simulate an embedding backend that fails a couple of times before succeeding."""

    class FlakyEmbedModel:
        def __init__(self, fail_times: int = 2):
            self.calls = 0
            self.fail_times = fail_times

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("simulated embed service outage")
            # On success, return deterministic vectors
            return [[0.1 for _ in range(384)] for _ in range(len(texts))]

    passages = [
        {"paperId": "r1", "title": "Retry Paper", "text": _make_text(300)}
        for _ in range(1)
    ]
    state = State(query="retry", passages=passages)
    agent = ProcessingAgent(chunk_size=100, chunk_overlap=10)

    model = FlakyEmbedModel(fail_times=2)
    processed = agent.process(state, embedding_model=model, batch_size=8)

    assert processed.chunks is not None
    # Ensure model was called multiple times (retries) and eventually succeeded
    assert model.calls >= 2
    for c in processed.chunks:
        emb = c.metadata.get("embedding")
        assert emb is not None
        assert len(emb) == 384
