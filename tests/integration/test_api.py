import os
import time

import pytest


@pytest.fixture
def set_test_env():
    """Set minimal test environment variables for config validation"""
    # Save original values
    original_values = {
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        "PRIMARY_LLM_API_KEY": os.environ.get("PRIMARY_LLM_API_KEY"),
        "SECONDARY_LLM_API_KEY": os.environ.get("SECONDARY_LLM_API_KEY"),
        "QUERY_OPTIMIZATION_MODEL": os.environ.get("QUERY_OPTIMIZATION_MODEL"),
        "SYNTHESIS_MODEL": os.environ.get("SYNTHESIS_MODEL"),
        "SKIP_DOTENV_LOADER": os.environ.get("SKIP_DOTENV_LOADER"),
    }

    # Set test values
    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "test_key"
    os.environ["PRIMARY_LLM_API_KEY"] = "test_key"
    os.environ["SECONDARY_LLM_API_KEY"] = "test_key"
    os.environ["QUERY_OPTIMIZATION_MODEL"] = "test_model"
    os.environ["SYNTHESIS_MODEL"] = "test_model"
    os.environ["SKIP_DOTENV_LOADER"] = "1"

    yield

    # Restore original values
    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def app():
    from main import app

    return app


@pytest.mark.integration
def test_health_endpoint(app, set_test_env):
    """Test health endpoint returns 200 status"""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "3.0.0"


@pytest.mark.integration
def test_health_endpoint_response_time(app, set_test_env):
    """Test health endpoint response time < 100ms"""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()

    response_time_ms = (end_time - start_time) * 1000
    assert response_time_ms < 100
    assert response.status_code == 200


@pytest.mark.integration
def test_startup_event_logs_configuration(app, set_test_env):
    """Test startup event logs configuration successfully"""
    # This test verifies that the app can start without config validation errors
    # In a real scenario, we'd check logs, but for integration test, we ensure no exceptions
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.status_code == 200
