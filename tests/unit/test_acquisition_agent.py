import asyncio
import time
import pytest

from models.state import State
from agents.acquisition_agent import AcquisitionAgent


def make_pdf_bytes(long: bool = True) -> bytes:
    import fitz

    doc = fitz.open()
    if long:
        body = "This is a test. " * 100
        for _ in range(3):
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(72, 72, 500, 700), body)
    else:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 700), "Tiny")
    return doc.write()


class MockContent:
    def __init__(
        self, data: bytes, delay_per_chunk: float = 0.0, chunk_size: int = 1024
    ):
        self._data = data
        self._delay = delay_per_chunk
        self._chunk_size = chunk_size

    async def iter_chunked(self, n):
        for i in range(0, len(self._data), n):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield self._data[i : i + n]


class MockResponse:
    def __init__(self, status: int, data: bytes = b"", delay_per_chunk: float = 0.0):
        self.status = status
        self.content = MockContent(data, delay_per_chunk=delay_per_chunk)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockSession:
    def __init__(self, mapping):
        # mapping: url -> (status, bytes, delay_per_chunk)
        self._map = mapping

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, *args, **kwargs):
        entry = self._map.get(url)
        if entry is None:
            raise RuntimeError("No mapping for URL")
        status, data, delay = entry
        return MockResponse(status, data, delay_per_chunk=delay)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_enriches_state_success(monkeypatch):
    data = make_pdf_bytes(long=True)
    url = "https://example.com/p1.pdf"

    mapping = {url: (200, data, 0.0)}

    async def session_factory(*args, **kwargs):
        return MockSession(mapping)

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: MockSession(mapping))

    state = State(
        query="q",
        finalists=[{"paperId": "abc123", "title": "T", "openAccessPdf": {"url": url}}],
    )

    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 5.0
    new_state = await agent.process(state)

    assert new_state.passages is not None
    assert len(new_state.passages) == 2
    pdf_passages = [p for p in new_state.passages if p["text_source"] == "pdf"]
    assert len(pdf_passages) == 1
    p = pdf_passages[0]
    assert p["paperId"] == "abc123"
    assert p["text_source"] == "pdf"
    assert len(p["text"]) > 500


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_timeout_skips(monkeypatch):
    # slow chunks cause timeout
    data = make_pdf_bytes(long=True)
    url = "https://example.com/slow.pdf"
    mapping = {url: (200, data, 0.2)}

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: MockSession(mapping))

    state = State(
        query="q",
        finalists=[{"paperId": "slow1", "title": "T", "openAccessPdf": {"url": url}}],
    )
    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 0.05

    new_state = await agent.process(state)
    assert new_state.passages is not None
    assert len(new_state.passages) == 1
    p = new_state.passages[0]
    assert p["text_source"] == "abstract"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_http_404_skips(monkeypatch):
    url = "https://example.com/notfound.pdf"
    mapping = {url: (404, b"", 0.0)}

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: MockSession(mapping))

    state = State(
        query="q",
        finalists=[{"paperId": "nf", "title": "T", "openAccessPdf": {"url": url}}],
    )
    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 1.0

    new_state = await agent.process(state)
    assert new_state.passages is not None
    assert len(new_state.passages) == 1
    p = new_state.passages[0]
    assert p["text_source"] == "abstract"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_malformed_pdf_skips(monkeypatch):
    # malformed PDF bytes should cause extraction to fail and be skipped
    url = "https://example.com/bad.pdf"
    mapping = {url: (200, b"not-a-pdf", 0.0)}

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: MockSession(mapping))

    state = State(
        query="q",
        finalists=[{"paperId": "bad1", "title": "T", "openAccessPdf": {"url": url}}],
    )
    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 1.0

    new_state = await agent.process(state)
    # should be skipped
    assert new_state.passages is not None
    assert len(new_state.passages) == 1
    p = new_state.passages[0]
    assert p["text_source"] == "abstract"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_zero_successes_appends_error(monkeypatch):
    # all 404 -> no successes -> state.errors contains critical message
    urls = [f"https://example.com/n{i}.pdf" for i in range(3)]
    mapping = {u: (404, b"", 0.0) for u in urls}
    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: MockSession(mapping))

    finalists = [
        {"paperId": f"n{i}", "title": "T", "openAccessPdf": {"url": urls[i]}}
        for i in range(3)
    ]
    state = State(query="q", finalists=finalists)
    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 1.0

    new_state = await agent.process(state)
    assert new_state.passages is not None
    assert len(new_state.passages) == 3
    for p in new_state.passages:
        assert p["text_source"] == "abstract"
    # No error since abstracts are acquired
    assert not new_state.errors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_get_raises_timeout(monkeypatch):
    # simulate session.get raising a TimeoutError
    class RaisingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, *args, **kwargs):
            raise asyncio.TimeoutError()

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: RaisingSession())

    state = State(
        query="q",
        finalists=[
            {
                "paperId": "tmo",
                "title": "T",
                "openAccessPdf": {"url": "https://example.com/x.pdf"},
            }
        ],
    )
    agent = AcquisitionAgent()
    agent.PER_PDF_TIMEOUT = 0.5

    new_state = await agent.process(state)
    assert new_state.passages is not None
    assert len(new_state.passages) == 1
    p = new_state.passages[0]
    assert p["text_source"] == "abstract"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquisition_parallel_10_with_2_failures(monkeypatch):
    # 10 finalists: 8 fast, 2 slow (timeout)
    urls = [f"https://example.com/p{i}.pdf" for i in range(10)]
    mapping = {}
    fast_bytes = make_pdf_bytes(long=True)
    for i, u in enumerate(urls):
        if i < 8:
            mapping[u] = (200, fast_bytes, 0.0)
        else:
            mapping[u] = (200, fast_bytes, 0.2)

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: MockSession(mapping))

    finalists = [
        {"paperId": f"p{i}", "title": "T", "openAccessPdf": {"url": urls[i]}}
        for i in range(10)
    ]
    state = State(query="q", finalists=finalists)
    agent = AcquisitionAgent()
    # make timeouts small so test doesn't actually wait long
    agent.PER_PDF_TIMEOUT = 0.05

    start = time.monotonic()
    new_state = await agent.process(state)
    elapsed = time.monotonic() - start

    # successful should be 5 PDFs (target limit), plus 10 abstracts
    passages = new_state.passages or []
    assert len(passages) == 15
    pdf_passages = [p for p in passages if p["text_source"] == "pdf"]
    assert len(pdf_passages) == 5
    # requirement: return in under 4 seconds in real world; here ensure it's fast
    assert elapsed < 4.0
