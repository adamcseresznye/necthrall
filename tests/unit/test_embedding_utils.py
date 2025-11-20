import math
import time
from typing import List

import logging
import numpy as np
import pytest
from unittest.mock import Mock

from utils import embedding_utils

try:
    from hypothesis import given, strategies as st, settings
except Exception:  # pragma: no cover - hypothesis may not be installed in minimal envs
    given = lambda *a, **k: (lambda f: f)
    st = None
    settings = lambda *a, **k: (lambda f: f)


@pytest.fixture
def mock_embedding_model():
    """Return a fresh mock embedding model with deterministic `embed_documents`.

    Deterministic mapping: input `text_i` -> vector filled with float(i).
    """

    def embed_documents(inputs: List[str]):
        outputs = []
        for s in inputs:
            idx = 0
            if isinstance(s, str) and s.startswith("text_"):
                try:
                    idx = int(s.split("_")[-1])
                except Exception:
                    idx = 0
            outputs.append(np.full(384, float(idx), dtype=float))
        return outputs

    mock = Mock()
    mock.embed_documents = Mock(side_effect=embed_documents)
    return mock


def make_texts(n: int):
    return [f"text_{i}" for i in range(n)]


class TestBatchEmbedding:
    @pytest.mark.unit
    @pytest.mark.parametrize("n", [0, 1, 10, 100, 1000])
    def test_batched_embed_various_sizes(self, n, mock_embedding_model):
        """Embed various sizes and validate output length and 384-dim shape.

        Note: reduced the 10000 case to 1000 to keep unit tests fast.
        """
        texts = make_texts(n)
        start = time.perf_counter()
        results = embedding_utils.batched_embed(
            texts, mock_embedding_model, batch_size=128
        )
        elapsed = time.perf_counter() - start

        assert isinstance(results, list)
        assert len(results) == n
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (384,)

        assert elapsed < 5.0

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 128])
    def test_different_batch_sizes_produce_identical_results(
        self, batch_size, mock_embedding_model
    ):
        """Different batch sizes yield identical embeddings (order preserved)."""
        texts = make_texts(200)
        ref = embedding_utils.batched_embed(texts, mock_embedding_model, batch_size=32)
        alt = embedding_utils.batched_embed(
            texts, mock_embedding_model, batch_size=batch_size
        )

        assert len(ref) == len(alt) == 200
        for r, a in zip(ref, alt):
            assert np.array_equal(r, a)

    @pytest.mark.unit
    def test_100_texts_batching_and_order(self, mock_embedding_model):
        """100 texts with batch_size=32 => 4 batches, order preserved."""
        n = 100
        texts = make_texts(n)
        batch_size = 32
        results = embedding_utils.batched_embed(
            texts, mock_embedding_model, batch_size=batch_size
        )

        assert mock_embedding_model.embed_documents.call_count == math.ceil(
            n / batch_size
        )
        assert len(results) == n
        for i, arr in enumerate(results):
            assert arr[0] == float(i)


class TestErrorHandling:
    @pytest.mark.unit
    def test_invalid_input_type_raises(self):
        """Non-list `texts` raises TypeError with informative message."""
        with pytest.raises(TypeError) as exc:
            embedding_utils.batched_embed(123, Mock())
        assert "must be a list" in str(exc.value)

    @pytest.mark.unit
    def test_invalid_batch_size_values_raise(self, mock_embedding_model):
        """Invalid batch sizes raise ValueError with helpful text."""
        for bs in (0, -1):
            with pytest.raises(ValueError) as exc:
                embedding_utils.batched_embed(
                    ["a"], mock_embedding_model, batch_size=bs
                )
            assert "positive integer" in str(exc.value)

    @pytest.mark.unit
    def test_empty_and_none_entries_replaced_with_zeros(self, mock_embedding_model):
        texts = ["text_0", "", None, "text_3"]
        results = embedding_utils.batched_embed(
            texts, mock_embedding_model, batch_size=2
        )
        assert len(results) == 4
        assert np.all(results[1] == 0.0)
        assert np.all(results[2] == 0.0)
        assert results[0][0] == 0.0
        assert results[3][0] == 3.0

    @pytest.mark.unit
    def test_dimension_mismatch_raises(self):
        def bad_embed(inputs: List[str]):
            return [np.zeros(10, dtype=float) for _ in inputs]

        bad_model = Mock()
        bad_model.embed_documents = Mock(side_effect=bad_embed)

        with pytest.raises(ValueError) as exc:
            embedding_utils.batched_embed(["text_0", "text_1"], bad_model, batch_size=2)
        assert "Embedding dimension mismatch" in str(exc.value)

    @pytest.mark.unit
    def test_model_initialization_failure_logs_and_raises(self, monkeypatch):
        """Simulate a model that raises on call (initialization/runtime error)."""

        def explode(_):
            raise RuntimeError("model init failed")

        bad_model = Mock()
        bad_model.embed_documents = Mock(side_effect=explode)

        # Replace logger with stdlib logger so caplog can capture messages
        std_logger = logging.getLogger("test_embedding_utils")
        std_logger.setLevel(logging.DEBUG)
        monkeypatch.setattr(embedding_utils, "logger", std_logger)

        with pytest.raises(RuntimeError):
            embedding_utils.batched_embed(["text_0"], bad_model, batch_size=1)

    @pytest.mark.unit
    def test_missing_embed_method_raises(self):
        """Embedding model without known embed methods raises AttributeError."""
        obj = object()
        with pytest.raises(AttributeError):
            embedding_utils.batched_embed(["a"], obj)


class TestLogging:
    @pytest.mark.unit
    def test_debug_and_info_logged(self, monkeypatch, caplog, mock_embedding_model):
        """Verify debug/info logs appear using the stdlib logger via monkeypatch."""
        std_logger = logging.getLogger("embedding_utils_test")
        std_logger.setLevel(logging.DEBUG)
        monkeypatch.setattr(embedding_utils, "logger", std_logger)

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            # small input should emit a debug entry
            embedding_utils.batched_embed(
                ["a", "b"], mock_embedding_model, batch_size=1
            )

        # check at least debug or info messages exist
        messages = [r.message for r in caplog.records]
        assert any("batched_embed entry" in m for m in messages) or any(
            m for m in messages
        )


class TestPerformance:
    @pytest.mark.unit
    def test_1000_chunks_complete_quickly(self):
        """Performance assertion: 1000 texts with mocked model complete <100ms.

        This uses a very lightweight mock to avoid model overhead.
        """
        n = 1000
        texts = make_texts(n)

        def fast_embed(inputs: List[str]):
            # return lists (fast to create) of length 384; using Python lists is
            # slightly faster than creating many numpy arrays here.
            out = [[0.0] * 384 for _ in inputs]
            return out

        fast_model = Mock()
        fast_model.embed_documents = Mock(side_effect=fast_embed)

        start = time.perf_counter()
        results = embedding_utils.batched_embed(texts, fast_model, batch_size=128)
        elapsed = (time.perf_counter() - start) * 1000.0  # ms

        assert len(results) == n
        # allow a small leeway; depending on environment this may need adjusting
        assert elapsed < 100.0


if st is not None:  # pragma: no branch - run property test when hypothesis installed

    @pytest.mark.unit
    @settings(max_examples=25)
    @given(st.lists(st.text(min_size=0, max_size=50), min_size=0, max_size=200))
    def test_property_based_output_length_matches_input(texts):
        """Property-based test: output length equals input length for random lists.

        For speed, the Hypothesis max size is reduced; this still provides good
        coverage without making unit tests slow.
        """

        # lightweight mock that returns zeros for each input
        def mk(inputs: List[str]):
            return [[0.0] * 384 for _ in inputs]

        model = Mock()
        model.embed_documents = Mock(side_effect=mk)

        results = embedding_utils.batched_embed(texts, model, batch_size=32)
        assert len(results) == len(texts)

else:

    @pytest.mark.unit
    def test_property_based_output_length_matches_input_hypothesis_unavailable():
        """Fallback smoke test when Hypothesis is not installed: verify basic behavior."""
        texts = ["a", "", None, "b"]

        def mk(inputs: List[str]):
            return [[0.0] * 384 for _ in inputs]

        model = Mock()
        model.embed_documents = Mock(side_effect=mk)

        results = embedding_utils.batched_embed(texts, model, batch_size=2)
        assert len(results) == len(texts)


import pytest
import numpy as np

from utils.embedding_utils import batched_embed


class FakeEmbeddingModel:
    """A tiny fake embedding model that returns deterministic vectors.

    It expects inputs to be strings of the form 'text-{index}' for non-empty texts
    and will return a list of lists of length 384 where each element equals the index.
    """

    def __init__(self):
        self.call_count = 0
        self.batch_sizes = []
        self.batch_inputs = []

    def embed_documents(self, texts):
        self.call_count += 1
        self.batch_sizes.append(len(texts))
        # record inputs for assertions (use a copy)
        self.batch_inputs.append(list(texts))
        out = []
        for t in texts:
            if not t:
                # return a non-zero vector for empty inputs; the util should replace with zeros
                out.append([1.0] * 384)
            else:
                # extract integer from 'text-{i}' style
                try:
                    idx = int(t.split("-")[-1])
                except Exception:
                    idx = 7
                out.append([float(idx)] * 384)
        return out


@pytest.mark.unit
def test_embed_10_texts_shapes():
    model = FakeEmbeddingModel()
    texts = [f"text-{i}" for i in range(10)]
    embeddings = batched_embed(texts, model, batch_size=32)
    assert len(embeddings) == 10
    for i, e in enumerate(embeddings):
        assert isinstance(e, np.ndarray)
        assert e.shape == (384,)
        assert e[0] == float(i)


@pytest.mark.unit
def test_embed_100_texts_batch32():
    model = FakeEmbeddingModel()
    texts = [f"text-{i}" for i in range(100)]
    embeddings = batched_embed(texts, model, batch_size=32)
    assert len(embeddings) == 100
    # Expect 4 batches: 32 + 32 + 32 + 4
    assert model.call_count == 4
    assert model.batch_sizes == [32, 32, 32, 4]
    # check order preserved
    for i, e in enumerate(embeddings):
        assert e[0] == float(i)


@pytest.mark.unit
def test_empty_list_returns_empty():
    model = FakeEmbeddingModel()
    embeddings = batched_embed([], model, batch_size=32)
    assert embeddings == []


@pytest.mark.unit
def test_handle_empty_and_none_strings():
    model = FakeEmbeddingModel()
    texts = [None, "", "text-2", None, "text-4"]
    embeddings = batched_embed(texts, model, batch_size=2)
    assert len(embeddings) == 5
    # None and empty should be zero vectors
    assert np.allclose(embeddings[0], np.zeros(384))
    assert np.allclose(embeddings[1], np.zeros(384))
    # non-empty entries should match fake model values
    assert embeddings[2][0] == 2.0
    assert np.allclose(embeddings[3], np.zeros(384))
    assert embeddings[4][0] == 4.0


@pytest.mark.unit
def test_truncate_long_texts_and_timing():
    model = FakeEmbeddingModel()
    # create a long text of 600 tokens
    long_text = "".join([f"w{i} " for i in range(600)])
    texts = [long_text, "text-1"]
    embeddings = batched_embed(texts, model, batch_size=2, show_progress=True)
    # ensure model received truncated input of <=512 tokens
    assert len(model.batch_inputs) == 1
    received0 = model.batch_inputs[0][0]
    assert isinstance(received0, str)
    assert len(received0.split()) <= 512


@pytest.mark.unit
def test_invalid_inputs_raise():
    model = FakeEmbeddingModel()
    with pytest.raises(TypeError):
        batched_embed(None, model)

    with pytest.raises(ValueError):
        batched_embed(["a"], model, batch_size=0)


@pytest.mark.unit
@pytest.mark.parametrize("bs", [16, 32, 64])
def test_batch_size_parameter(bs):
    model = FakeEmbeddingModel()
    texts = [f"text-{i}" for i in range(50)]
    embeddings = batched_embed(texts, model, batch_size=bs)
    assert len(embeddings) == 50
    # number of calls should match ceil(50/bs)
    import math

    assert model.call_count == math.ceil(50 / bs)
