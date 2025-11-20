import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_init_success(monkeypatch):
    """Successful initialization stores model on app.state."""

    class FakeModel:
        def __init__(self, model_name: str, device: str):
            self.model_name = model_name
            self.device = device
            self.embed_dim = 384

        def embed_documents(self, texts):
            return [[0.0] * 384 for _ in texts]

    import sys

    monkeypatch.setitem(
        sys.modules,
        "llama_index.embeddings",
        SimpleNamespace(HuggingFaceEmbedding=FakeModel),
    )

    from fastapi import FastAPI

    from config.embedding_config import init_embedding

    app = FastAPI()

    asyncio.get_event_loop().run_until_complete(init_embedding(app))

    assert hasattr(app.state, "embedding_model")
    model = app.state.embedding_model
    assert isinstance(model, FakeModel)


@pytest.mark.unit
def test_embedding_dimension_and_device(monkeypatch):
    """Validate the embedding dimension is 384 and device is cpu."""

    class FakeModel:
        def __init__(self, model_name: str, device: str):
            self.model_name = model_name
            self.device = device

        def embed_documents(self, texts):
            return [[1.0] * 384 for _ in texts]

        @property
        def embed_dim(self):
            return 384

    import sys

    monkeypatch.setitem(
        sys.modules,
        "llama_index.embeddings",
        SimpleNamespace(HuggingFaceEmbedding=FakeModel),
    )

    from fastapi import FastAPI
    from config.embedding_config import init_embedding, get_embedding_model

    app = FastAPI()
    asyncio.get_event_loop().run_until_complete(init_embedding(app))

    model = get_embedding_model(app)
    assert getattr(model, "device", "cpu") == "cpu"
    # check embed_documents returns 384-d vector
    vec = model.embed_documents(["hello"])[0]
    assert len(vec) == 384


@pytest.mark.unit
def test_init_failure_network(monkeypatch):
    """Simulate network/download failure and ensure initialization raises RuntimeError."""

    def broken_init(model_name: str, device: str):
        raise OSError("Network error: failed to download model")

    import sys

    monkeypatch.setitem(
        sys.modules,
        "llama_index.embeddings",
        SimpleNamespace(HuggingFaceEmbedding=broken_init),
    )

    from fastapi import FastAPI
    from config.embedding_config import init_embedding

    app = FastAPI()

    with pytest.raises(RuntimeError):
        asyncio.get_event_loop().run_until_complete(init_embedding(app))


@pytest.mark.unit
def test_retry_logic(monkeypatch):
    """Simulate transient failures on first two attempts, succeed on third."""

    class FlakyModel:
        counter = 0

        def __init__(self, model_name: str, device: str):
            FlakyModel.counter += 1
            # fail first two times to trigger retry/backoff
            if FlakyModel.counter < 3:
                raise OSError("transient network error")

            self.model_name = model_name
            self.device = device
            self.embed_dim = 384

        def embed_documents(self, texts):
            return [[0.0] * 384 for _ in texts]

    import sys

    monkeypatch.setitem(
        sys.modules,
        "llama_index.embeddings",
        SimpleNamespace(HuggingFaceEmbedding=FlakyModel),
    )

    from fastapi import FastAPI
    from config.embedding_config import init_embedding

    app = FastAPI()
    asyncio.get_event_loop().run_until_complete(init_embedding(app))

    assert hasattr(app.state, "embedding_model")
    model = app.state.embedding_model
    assert isinstance(model, FlakyModel)


@pytest.mark.unit
def test_memory_logging(monkeypatch, caplog):
    """Verify memory footprint is logged when psutil is available."""

    class FakeModel:
        def __init__(self, model_name: str, device: str):
            self.model_name = model_name
            self.device = device
            self.embed_dim = 384

        def embed_documents(self, texts):
            return [[1.0] * 384 for _ in texts]

    class FakeMemInfo:
        def __init__(self, rss):
            self.rss = rss

    class FakeProcess:
        def memory_info(self):
            return FakeMemInfo(rss=50 * 1024 * 1024)

    def fake_Process():
        return FakeProcess()

    import sys

    monkeypatch.setitem(
        sys.modules,
        "llama_index.embeddings",
        SimpleNamespace(HuggingFaceEmbedding=FakeModel),
    )
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=fake_Process))

    from fastapi import FastAPI
    from config.embedding_config import init_embedding

    app = FastAPI()

    # Capture logger.info calls from the embedding_config module (loguru).
    import config.embedding_config as ec

    captured: list[str] = []

    def fake_info(msg, *args, **kwargs):
        try:
            captured.append(msg % args if args else msg)
        except Exception:
            captured.append(str(msg))

    monkeypatch.setattr(ec, "logger", ec.logger)
    monkeypatch.setattr(ec.logger, "info", fake_info)

    asyncio.get_event_loop().run_until_complete(init_embedding(app))

    joined = "\n".join(captured).lower()
    assert ("memory" in joined) or (
        "mb" in joined
    ), f"Expected memory info to be logged, got: {joined}"
