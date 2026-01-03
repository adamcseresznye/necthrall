"""Unit tests for config/embedding_config.py.

These tests verify the ONNX embedding model initialization and error handling.
They use mocks to avoid loading the actual ONNX model (Windows DLL issues).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class MockONNXEmbeddingModel:
    """Mock ONNXEmbeddingModel for testing."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embed_dim = 384

    def get_text_embedding_batch(self, texts):
        return [[0.0] * 384 for _ in texts]


@pytest.mark.unit
def test_init_embedding_success(monkeypatch):
    """Successful initialization stores ONNXEmbeddingModel on app.state."""
    import config.embedding_config as ec

    # Reset the lazy import state
    monkeypatch.setattr(ec, "_onnx_available", None)
    monkeypatch.setattr(ec, "_init_onnx", None)

    # Mock the lazy import function to return our mock
    def mock_lazy_import():
        return True, lambda: MockONNXEmbeddingModel()

    monkeypatch.setattr(ec, "_lazy_import_onnx", mock_lazy_import)

    from fastapi import FastAPI

    app = FastAPI()
    model = asyncio.get_event_loop().run_until_complete(ec.init_embedding())
    app.state.embedding_model = model

    assert hasattr(app.state, "embedding_model")
    model = app.state.embedding_model
    assert model is not None
    assert type(model).__name__ == "MockONNXEmbeddingModel"


@pytest.mark.unit
def test_embedding_model_has_required_methods(monkeypatch):
    """Validate the embedding model has get_text_embedding_batch method."""
    import config.embedding_config as ec

    # Reset and mock lazy import
    monkeypatch.setattr(ec, "_onnx_available", None)
    monkeypatch.setattr(ec, "_init_onnx", None)

    def mock_lazy_import():
        return True, lambda: MockONNXEmbeddingModel()

    monkeypatch.setattr(ec, "_lazy_import_onnx", mock_lazy_import)

    from fastapi import FastAPI

    app = FastAPI()
    model = asyncio.get_event_loop().run_until_complete(ec.init_embedding())
    app.state.embedding_model = model

    model = ec.get_embedding_model(app)
    assert hasattr(model, "get_text_embedding_batch")
    assert callable(model.get_text_embedding_batch)

    # Verify embeddings have correct dimension
    embeddings = model.get_text_embedding_batch(["test"])
    assert len(embeddings[0]) == 384


@pytest.mark.unit
def test_init_embedding_missing_onnxruntime(monkeypatch):
    """When onnxruntime is missing, embedding_model should be None but app starts."""
    import config.embedding_config as ec

    # Reset and mock lazy import to simulate missing onnxruntime
    monkeypatch.setattr(ec, "_onnx_available", None)
    monkeypatch.setattr(ec, "_init_onnx", None)

    def mock_lazy_import():
        return False, None

    monkeypatch.setattr(ec, "_lazy_import_onnx", mock_lazy_import)

    from fastapi import FastAPI

    app = FastAPI()
    model = asyncio.get_event_loop().run_until_complete(ec.init_embedding())
    app.state.embedding_model = model

    assert hasattr(app.state, "embedding_model")
    assert app.state.embedding_model is None


@pytest.mark.unit
def test_init_embedding_missing_model_file(monkeypatch):
    """When model file is missing, embedding_model should be None but app starts."""
    import config.embedding_config as ec

    def raise_runtime_error():
        raise RuntimeError("Model missing: /path/to/model.onnx. Run setup_onnx.py.")

    # Reset and mock lazy import
    monkeypatch.setattr(ec, "_onnx_available", None)
    monkeypatch.setattr(ec, "_init_onnx", None)

    def mock_lazy_import():
        return True, raise_runtime_error

    monkeypatch.setattr(ec, "_lazy_import_onnx", mock_lazy_import)

    from fastapi import FastAPI

    app = FastAPI()
    model = asyncio.get_event_loop().run_until_complete(ec.init_embedding())
    app.state.embedding_model = model

    assert hasattr(app.state, "embedding_model")
    assert app.state.embedding_model is None


@pytest.mark.unit
def test_get_embedding_model_not_initialized():
    """get_embedding_model raises RuntimeError if called before init_embedding."""
    from fastapi import FastAPI

    import config.embedding_config as ec

    app = FastAPI()
    # Don't call init_embedding

    with pytest.raises(RuntimeError, match="not initialized"):
        ec.get_embedding_model(app)


@pytest.mark.unit
def test_get_embedding_model_returns_none_when_failed(monkeypatch):
    """get_embedding_model returns None if init failed (graceful degradation)."""
    import config.embedding_config as ec

    # Reset and mock lazy import
    monkeypatch.setattr(ec, "_onnx_available", None)
    monkeypatch.setattr(ec, "_init_onnx", None)

    def mock_lazy_import():
        return False, None

    monkeypatch.setattr(ec, "_lazy_import_onnx", mock_lazy_import)

    from fastapi import FastAPI

    app = FastAPI()
    model = asyncio.get_event_loop().run_until_complete(ec.init_embedding())
    app.state.embedding_model = model

    model = ec.get_embedding_model(app)
    assert model is None


@pytest.mark.unit
def test_memory_logging(monkeypatch, caplog):
    """Verify memory footprint is logged when psutil is available."""
    import sys

    import config.embedding_config as ec

    class FakeMemInfo:
        def __init__(self, rss):
            self.rss = rss

    class FakeProcess:
        def memory_info(self):
            return FakeMemInfo(rss=50 * 1024 * 1024)

    def fake_Process():
        return FakeProcess()

    # Mock psutil
    fake_psutil = SimpleNamespace(Process=fake_Process)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    # Reset and mock lazy import
    monkeypatch.setattr(ec, "_onnx_available", None)
    monkeypatch.setattr(ec, "_init_onnx", None)

    def mock_lazy_import():
        return True, lambda: MockONNXEmbeddingModel()

    monkeypatch.setattr(ec, "_lazy_import_onnx", mock_lazy_import)

    from fastapi import FastAPI

    app = FastAPI()

    # Capture logger.info calls
    captured = []

    def fake_info(msg, *args, **kwargs):
        try:
            captured.append(msg % args if args else str(msg))
        except Exception:
            captured.append(str(msg))

    original_info = ec.logger.info
    monkeypatch.setattr(ec.logger, "info", fake_info)

    model = asyncio.get_event_loop().run_until_complete(ec.init_embedding())
    app.state.embedding_model = model

    joined = "\n".join(captured).lower()
    assert ("ram" in joined) or (
        "mb" in joined
    ), f"Expected memory info to be logged, got: {joined}"
        "mb" in joined
    ), f"Expected memory info to be logged, got: {joined}"
