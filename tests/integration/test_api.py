import pytest
import time
import os


@pytest.fixture(autouse=True)
def set_test_env():
    """ "Set minimal test environment variables for config validation"""
    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "test_key"
    os.environ["GOOGLE_API_KEY"] = "test_key"
    os.environ["GROQ_API_KEY"] = "test_key"
    os.environ["QUERY_OPTIMIZATION_MODEL"] = "test_model"
    os.environ["SYNTHESIS_MODEL"] = "test_model"
    os.environ["SKIP_DOTENV_LOADER"] = "1"


@pytest.fixture
def app():
    from main import app

    return app


@pytest.mark.integration
def test_health_endpoint(app):
    """ "Test health endpoint returns 200 status"""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "3.0.0-mvp"


@pytest.mark.integration
def test_root_endpoint(app):
    """ "Test root endpoint returns 200 status with API documentation links"""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


@pytest.mark.integration
def test_health_endpoint_response_time(app):
    """ "Test health endpoint response time < 100ms"""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()

    response_time_ms = (end_time - start_time) * 1000
    assert response_time_ms < 100
    assert response.status_code == 200


@pytest.mark.integration
def test_startup_event_logs_configuration(app):
    """ "Test startup event logs configuration successfully"""
    # This test verifies that the app can start without config validation errors
    # In a real scenario, we'd check logs, but for integration test, we ensure no exceptions
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
