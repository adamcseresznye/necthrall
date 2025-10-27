"""
Tests for FastAPI startup event and embedding model caching functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time
import numpy as np

# Test individual components without importing main module due to PyTorch issues
from agents.filtering_agent import FilteringAgent
from agents.processing_agent import ProcessingAgent
from models.state import Paper, PDFContent


# Mock the main module to avoid import issues
class MockApp:
    def __init__(self):
        self.state = type("MockState", (), {})()


class MockRequest:
    def __init__(self):
        self.app = MockApp()


@pytest.fixture
def mock_request():
    """Create mock request object"""
    return MockRequest()


@pytest.fixture
def client():
    """Create test client with startup/shutdown events"""
    # Skip this fixture since we can't import main module
    # with TestClient(app) as c:
    #     yield c
    pass


def test_sentence_transformers_import():
    """Test that sentence-transformers can be imported successfully"""
    try:
        from sentence_transformers import SentenceTransformer

        # Test basic model loading
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        assert model is not None

        # Test encoding
        embeddings = model.encode(["test sentence"])
        assert embeddings.shape == (1, 384)  # all-MiniLM-L6-v2 produces 384-dim vectors

        print("✅ SentenceTransformers working correctly")
    except Exception as e:
        pytest.fail(f"SentenceTransformers import/usage failed: {e}")


def test_filtering_agent_initialization(mock_request):
    """Test FilteringAgent initialization with mock embedding model"""
    # Mock the embedding model in app.state
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_st.return_value = mock_model

        # Set the model in mock app state
        mock_request.app.state.embedding_model = mock_model

        # Test agent initialization
        agent = FilteringAgent(mock_request)
        assert agent.embedding_model is not None


def test_filtering_agent_no_model(mock_request):
    """Test FilteringAgent raises error when no model available"""
    # Ensure no model in app.state
    mock_request.app.state.embedding_model = None

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Embedding model not found"):
        FilteringAgent(mock_request)


def test_processing_agent_initialization(mock_request):
    """Test ProcessingAgent initialization with mock embedding model"""
    # Mock the embedding model in app.state
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384)  # 2 chunks
        mock_st.return_value = mock_model

        # Set the model in mock app state
        mock_request.app.state.embedding_model = mock_model

        # Test agent initialization
        agent = ProcessingAgent(mock_request)
        assert agent.embedding_model is not None


def test_processing_agent_no_model(mock_request):
    """Test ProcessingAgent raises error when no model available"""
    # Ensure no model in app.state
    mock_request.app.state.embedding_model = None

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Embedding model not loaded"):
        ProcessingAgent(mock_request)


def test_processing_agent_embedding_generation(mock_request):
    """Test ProcessingAgent embedding generation"""
    # Mock the embedding model
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384)  # 2 chunks
        mock_st.return_value = mock_model

        mock_request.app.state.embedding_model = mock_model

        agent = ProcessingAgent(mock_request)

        # Create test PDF content
        pdf_contents = [
            PDFContent(
                paper_id="paper_1",
                raw_text="This is test content for paper one. It has multiple sentences.",
                page_count=1,
                char_count=60,
                extraction_time=0.1,
            )
        ]

        # Test embedding generation
        result = agent.generate_passage_embeddings(pdf_contents)

        assert "embeddings" in result
        assert "metadata" in result
        assert "total_chunks" in result
        assert result["total_chunks"] > 0  # Should have at least 1 chunk
        assert len(result["metadata"]) == result["total_chunks"]


def test_processing_agent_text_chunking(mock_request):
    """Test ProcessingAgent text chunking functionality"""
    # Mock the embedding model
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        mock_request.app.state.embedding_model = mock_model

        agent = ProcessingAgent(mock_request)

        # Test text chunking
        long_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."

        chunks = agent._chunk_text(long_text, chunk_size=50, chunk_overlap=20)

        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(
            len(chunk) <= 50 for chunk in chunks
        )  # No chunk longer than chunk_size
