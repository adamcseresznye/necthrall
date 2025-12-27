"""Integration tests for the /query API endpoint.

Tests the FastAPI endpoint that exposes the full 10-stage pipeline.
Validates the response schema critical for Frontend.

Test Coverage:
    - Happy path: Valid POST request returns 200 with expected JSON structure
    - Validation: Empty query returns 422 (Pydantic validation error)
    - Error handling: Pipeline failure returns 500 with clean error message

Run these tests with:
    pytest tests/integration/test_api_endpoint.py -v

Windows Note:
    On Windows, these tests may be skipped due to a torch DLL loading issue.
    This occurs when pytest's conftest.py loads certain libraries before main.py,
    causing torch's c10.dll to fail initialization. These tests work correctly:
    - In CI/CD environments (Linux/Docker)
    - When running the app directly (python main.py)
    - In non-Windows environments

    To run these tests on Windows, try running them in isolation:
        python -m pytest tests/integration/test_api_endpoint.py -v --forked

    Or run the main.py directly and test with curl/httpie.
"""

from __future__ import annotations

import os
import sys

# CRITICAL: Set environment variables BEFORE any imports that might use parallel processing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import TYPE_CHECKING, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Lazy imports - only for type hints, actual imports happen in fixtures
if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore, TextNode

# Check if we can import main without DLL errors (Windows issue)
_main_import_error = None
try:
    # Set environment first
    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = os.environ.get(
        "SEMANTIC_SCHOLAR_API_KEY", "test_key"
    )
    os.environ["PRIMARY_LLM_API_KEY"] = os.environ.get(
        "PRIMARY_LLM_API_KEY", "test_key"
    )
    os.environ["SECONDARY_LLM_API_KEY"] = os.environ.get(
        "SECONDARY_LLM_API_KEY", "test_key"
    )
    os.environ["QUERY_OPTIMIZATION_MODEL"] = os.environ.get(
        "QUERY_OPTIMIZATION_MODEL", "test_model"
    )
    os.environ["SYNTHESIS_MODEL"] = os.environ.get("SYNTHESIS_MODEL", "test_model")
    os.environ["SKIP_DOTENV_LOADER"] = "1"
    from main import app as _app  # noqa: F401 - Test import
except OSError as e:
    if "DLL" in str(e):
        _main_import_error = str(e)
    else:
        raise

# Skip all tests if we can't import main due to Windows DLL issues
pytestmark = pytest.mark.skipif(
    _main_import_error is not None,
    reason=f"Windows DLL error prevents main.py import: {_main_import_error}",
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def set_test_env():
    """Set minimal test environment variables for config validation."""
    original_values = {
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        "PRIMARY_LLM_API_KEY": os.environ.get("PRIMARY_LLM_API_KEY"),
        "SECONDARY_LLM_API_KEY": os.environ.get("SECONDARY_LLM_API_KEY"),
        "QUERY_OPTIMIZATION_MODEL": os.environ.get("QUERY_OPTIMIZATION_MODEL"),
        "SYNTHESIS_MODEL": os.environ.get("SYNTHESIS_MODEL"),
        "SKIP_DOTENV_LOADER": os.environ.get("SKIP_DOTENV_LOADER"),
    }

    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "test_key"
    os.environ["PRIMARY_LLM_API_KEY"] = "test_key"
    os.environ["SECONDARY_LLM_API_KEY"] = "test_key"
    os.environ["QUERY_OPTIMIZATION_MODEL"] = "test_model"
    os.environ["SYNTHESIS_MODEL"] = "test_model"
    os.environ["SKIP_DOTENV_LOADER"] = "1"

    yield

    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def app(set_test_env):
    """Get the FastAPI app instance."""
    from main import app

    return app


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def mock_pipeline_result():
    """Create a mock PipelineResult with realistic data."""
    from llama_index.core.schema import NodeWithScore, TextNode

    from services.query_service import PipelineResult

    # Create mock passages with NodeWithScore objects
    mock_passages: List[NodeWithScore] = []
    passage_texts = [
        "Intermittent fasting has been shown to significantly reduce blood pressure, with systolic pressure decreasing by an average of 8.5 mmHg in clinical trials.",
        "Time-restricted eating improves cardiac function and enhances left ventricular ejection fraction by approximately 4.2%.",
        "Long-term adherence to fasting protocols is associated with reduced atherosclerotic burden and improved lipid profiles.",
    ]

    for idx, text in enumerate(passage_texts):
        node = TextNode(
            text=text,
            metadata={
                "paper_id": f"paper_{idx + 1}",
                "paper_title": f"Cardiovascular Effects Study {idx + 1}",
                "section_name": "Results",
                "citation_count": 100 + idx * 50,
            },
        )
        mock_passages.append(NodeWithScore(node=node, score=0.95 - idx * 0.1))

    return PipelineResult(
        query="What are the cardiovascular effects of intermittent fasting?",
        optimized_queries={
            "primary": "cardiovascular effects intermittent fasting",
            "broad": "fasting heart health",
            "alternative": "time-restricted eating cardiovascular",
            "final_rephrase": "cardiovascular effects of intermittent fasting",
        },
        quality_gate={"passed": True, "metrics": {}},
        finalists=[
            {
                "paperId": "paper_1",
                "title": "Cardiovascular Effects Study 1",
                "citationCount": 150,
                "year": 2023,
            },
            {
                "paperId": "paper_2",
                "title": "Cardiovascular Effects Study 2",
                "citationCount": 200,
                "year": 2022,
            },
        ],
        execution_time=5.5,
        timing_breakdown={
            "query_optimization": 0.5,
            "semantic_scholar_search": 1.0,
            "quality_gate": 0.1,
            "composite_scoring": 0.2,
            "pdf_acquisition": 1.5,
            "processing": 1.0,
            "retrieval": 0.5,
            "reranking": 0.5,
            "synthesis": 0.15,
            "verification": 0.05,
        },
        success=True,
        passages=mock_passages,
        answer=(
            "Intermittent fasting has been shown to reduce blood pressure significantly [1], "
            "improve cardiac function [2], and reduce atherosclerotic burden [3]."
        ),
        citation_verification={
            "valid": True,
            "citations_found": [1, 2, 3],
            "invalid_citations": [],
        },
    )


@pytest.fixture
def mock_failed_pipeline_result():
    """Create a mock PipelineResult for a failed pipeline."""
    from services.query_service import PipelineResult

    return PipelineResult(
        query="test query",
        optimized_queries={},
        quality_gate={},
        finalists=[],
        execution_time=1.0,
        timing_breakdown={"query_optimization": 0.5},
        success=False,
        error_message="LLM API connection failed",
        error_stage="synthesis",
        passages=[],
        answer=None,
    )


# ============================================================================
# Happy Path Tests
# ============================================================================


@pytest.mark.integration
def test_query_endpoint_returns_200_with_valid_request(
    client, app, mock_pipeline_result, set_test_env
):
    """Test valid POST request returns 200 and expected JSON structure.

    Validates:
        - Status code is 200
        - Response contains all required fields: answer, citations, finalists, execution_time, timing_breakdown
        - Citations have correct structure (id, text, metadata)
    """
    # Mock the query service
    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(return_value=mock_pipeline_result)

        response = client.post(
            "/query",
            json={
                "query": "What are the cardiovascular effects of intermittent fasting?"
            },
        )

    # Assertions
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Check required top-level fields
    assert "answer" in data, "Response should contain 'answer' field"
    assert "citations" in data, "Response should contain 'citations' field"
    assert "finalists" in data, "Response should contain 'finalists' field"
    assert "execution_time" in data, "Response should contain 'execution_time' field"
    assert (
        "timing_breakdown" in data
    ), "Response should contain 'timing_breakdown' field"

    # Validate answer
    assert isinstance(data["answer"], str), "answer should be a string"
    assert len(data["answer"]) > 0, "answer should not be empty"
    assert "[1]" in data["answer"], "answer should contain citation [1]"

    # Validate citations structure
    assert isinstance(data["citations"], list), "citations should be a list"
    assert len(data["citations"]) > 0, "citations should not be empty"

    for citation in data["citations"]:
        assert "id" in citation, "citation should have 'id' field"
        assert "text" in citation, "citation should have 'text' field"
        assert "metadata" in citation, "citation should have 'metadata' field"
        assert isinstance(citation["id"], int), "citation id should be an integer"
        assert citation["id"] >= 1, "citation id should be >= 1"
        assert isinstance(citation["text"], str), "citation text should be a string"
        assert isinstance(
            citation["metadata"], dict
        ), "citation metadata should be a dict"

    # Validate finalists
    assert isinstance(data["finalists"], list), "finalists should be a list"
    assert len(data["finalists"]) > 0, "finalists should not be empty"

    # Validate execution_time
    assert isinstance(
        data["execution_time"], (int, float)
    ), "execution_time should be a number"
    assert data["execution_time"] > 0, "execution_time should be positive"

    # Validate timing_breakdown
    assert isinstance(
        data["timing_breakdown"], dict
    ), "timing_breakdown should be a dict"
    assert len(data["timing_breakdown"]) > 0, "timing_breakdown should not be empty"


@pytest.mark.integration
def test_query_endpoint_citation_metadata_includes_score(
    client, app, mock_pipeline_result, set_test_env
):
    """Test that citation metadata includes the reranking score for frontend use."""
    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(return_value=mock_pipeline_result)

        response = client.post(
            "/query",
            json={"query": "test query"},
        )

    assert response.status_code == 200
    data = response.json()

    # Check that score is in metadata
    for citation in data["citations"]:
        assert (
            "score" in citation["metadata"]
        ), "citation metadata should include 'score'"
        assert isinstance(
            citation["metadata"]["score"], (int, float)
        ), "score should be a number"


# ============================================================================
# Validation Tests
# ============================================================================


@pytest.mark.integration
def test_query_endpoint_empty_query_returns_422(client, set_test_env):
    """Test that an empty query returns 422 (Pydantic validation error).

    FastAPI/Pydantic validates min_length=1 and returns 422 Unprocessable Entity.
    """
    response = client.post("/query", json={"query": ""})

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    data = response.json()
    assert "detail" in data, "Response should contain 'detail' field"


@pytest.mark.integration
def test_query_endpoint_missing_query_field_returns_422(client, set_test_env):
    """Test that missing query field returns 422."""
    response = client.post("/query", json={})

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


@pytest.mark.integration
def test_query_endpoint_query_too_long_returns_422(client, set_test_env):
    """Test that a query exceeding max_length returns 422."""
    long_query = "a" * 501  # max_length is 500

    response = client.post("/query", json={"query": long_query})

    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
def test_query_endpoint_pipeline_failure_returns_500(
    client, app, mock_failed_pipeline_result, set_test_env
):
    """Test that pipeline failure returns HTTP 500 with clean error message."""
    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(return_value=mock_failed_pipeline_result)

        response = client.post(
            "/query",
            json={"query": "test query"},
        )

    assert response.status_code == 500, f"Expected 500, got {response.status_code}"

    data = response.json()
    assert "detail" in data, "Response should contain 'detail' field"
    assert "synthesis" in data["detail"], "Error should mention the failed stage"


@pytest.mark.integration
def test_query_endpoint_unexpected_exception_returns_500(client, app, set_test_env):
    """Test that unexpected exceptions return HTTP 500 with generic message."""
    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(
            side_effect=RuntimeError("Unexpected database error")
        )

        response = client.post(
            "/query",
            json={"query": "test query"},
        )

    assert response.status_code == 500, f"Expected 500, got {response.status_code}"

    data = response.json()
    assert "detail" in data, "Response should contain 'detail' field"
    # Should not expose internal error details to client
    assert "An unexpected error occurred" in data["detail"]


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.integration
def test_query_endpoint_no_answer_returns_fallback_message(client, app, set_test_env):
    """Test that when answer is None, a fallback message is returned."""
    from models.state import Passage
    from services.query_service import PipelineResult

    # Create result with no answer
    passage = Passage(paper_id="p1", text="Some passage text", score=0.9)
    result_no_answer = PipelineResult(
        query="test query",
        optimized_queries={
            "primary": "test",
            "broad": "test",
            "alternative": "test",
            "final_rephrase": "test",
        },
        quality_gate={"passed": True},
        finalists=[{"paperId": "p1", "title": "Test"}],
        execution_time=1.0,
        timing_breakdown={"query_optimization": 0.5},
        success=True,
        passages=[passage],
        answer=None,  # No answer generated
    )

    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(return_value=result_no_answer)

        response = client.post("/query", json={"query": "test query"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "No answer could be generated from the available sources."


@pytest.mark.integration
def test_query_endpoint_empty_passages_returns_empty_citations(
    client, app, set_test_env
):
    """Test that empty passages result in empty citations list."""
    from services.query_service import PipelineResult

    result_no_passages = PipelineResult(
        query="test query",
        optimized_queries={
            "primary": "test",
            "broad": "test",
            "alternative": "test",
            "final_rephrase": "test",
        },
        quality_gate={"passed": True},
        finalists=[{"paperId": "p1", "title": "Test"}],
        execution_time=1.0,
        timing_breakdown={"query_optimization": 0.5},
        success=True,
        passages=[],  # No passages
        answer="I cannot answer this based on the provided sources.",
    )

    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(return_value=result_no_passages)

        response = client.post("/query", json={"query": "test query"})

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == [], "citations should be empty when no passages"


@pytest.mark.integration
def test_query_endpoint_response_time_reasonable(
    client, app, mock_pipeline_result, set_test_env
):
    """Test that endpoint response time is reasonable (mocked pipeline)."""
    import time

    with patch.object(app.state, "query_service", create=True) as mock_service:
        mock_service.process_query = AsyncMock(return_value=mock_pipeline_result)

        start = time.time()
        response = client.post("/query", json={"query": "test query"})
        elapsed = time.time() - start

    assert response.status_code == 200
    # With mocked pipeline, response should be fast
    assert elapsed < 1.0, f"Response took {elapsed:.2f}s, expected < 1s"
    # With mocked pipeline, response should be fast
    assert elapsed < 1.0, f"Response took {elapsed:.2f}s, expected < 1s"
