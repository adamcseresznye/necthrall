import os
from unittest.mock import patch

import pytest

from config.config import Settings, get_settings


@pytest.mark.unit
def test_settings_defaults():
    """Test that settings have expected default values."""
    # Clear cache to ensure fresh settings
    get_settings.cache_clear()

    # Patch env to be empty to test defaults
    with patch.dict(os.environ, {}, clear=True):
        # Pass _env_file=None to ignore .env file
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == "INFO"
        assert settings.TIMEOUT == 30
        assert settings.RAG_RETRIEVAL_TOP_K == 50
        assert settings.SEMANTIC_SCHOLAR_API_KEY is None


@pytest.mark.unit
def test_settings_env_override():
    """Test that environment variables override defaults."""
    env = {
        "LOG_LEVEL": "DEBUG",
        "TIMEOUT": "60",
        "SEMANTIC_SCHOLAR_API_KEY": "test_key",
        "RAG_RETRIEVAL_TOP_K": "100",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings()
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.TIMEOUT == 60
        assert settings.SEMANTIC_SCHOLAR_API_KEY == "test_key"
        assert settings.RAG_RETRIEVAL_TOP_K == 100


@pytest.mark.unit
def test_validate_keys_logging(caplog):
    """Test that validate_keys logs warnings for missing keys."""
    # Clear cache
    get_settings.cache_clear()

    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
        settings.validate_keys()

        assert "SEMANTIC_SCHOLAR_API_KEY is not set" in caplog.text
        assert "PRIMARY_LLM_API_KEY is not set" in caplog.text


@pytest.mark.unit
def test_validate_keys_no_warning_when_set(caplog):
    """Test that validate_keys does not log warnings when keys are present."""
    env = {"SEMANTIC_SCHOLAR_API_KEY": "present", "PRIMARY_LLM_API_KEY": "present"}
    with patch.dict(os.environ, env, clear=True):
        settings = Settings()
        settings.validate_keys()

        assert "SEMANTIC_SCHOLAR_API_KEY is not set" not in caplog.text
        assert "PRIMARY_LLM_API_KEY is not set" not in caplog.text

        assert "SEMANTIC_SCHOLAR_API_KEY is not set" not in caplog.text
        assert "PRIMARY_LLM_API_KEY is not set" not in caplog.text


@pytest.mark.unit
def test_settings_has_research_constants():
    s = Settings(NICEGUI_STORAGE_SECRET="x")
    assert s.RESEARCH_MAX_ROUNDS == 3
    assert s.RESEARCH_PDF_TARGETS == {1: 3, 2: 2, 3: 1}
    assert s.ACQUISITION_TARGET_PDF_COUNT == 5
    assert s.ACQUISITION_PER_PDF_TIMEOUT == 30.0
    assert s.ACQUISITION_CHUNK_SIZE == 32768
    assert s.DISCOVERY_DEFAULT_WEIGHTS["relevance"] == 0.60
    assert s.DISCOVERY_NEWS_WEIGHTS["recency"] == 0.50
    assert s.DISCOVERY_FOUNDATIONAL_WEIGHTS["authority"] == 0.60


@pytest.mark.unit
def test_settings_no_rag_retrieval_mode():
    """Ensure the defunct RAG_RETRIEVAL_MODE and RAG_RERANK_TOP_K fields are gone."""
    s = Settings(NICEGUI_STORAGE_SECRET="x")
    assert not hasattr(s, "RAG_RETRIEVAL_MODE")
    assert not hasattr(s, "RAG_RERANK_TOP_K")


@pytest.mark.unit
def test_acquisition_agent_reads_from_settings():
    from agents.acquisition_agent import AcquisitionAgent
    from config.config import Settings

    custom = Settings(NICEGUI_STORAGE_SECRET="x", ACQUISITION_TARGET_PDF_COUNT=7)
    agent = AcquisitionAgent(settings=custom)
    assert agent._target_pdf_count == 7


@pytest.mark.unit
def test_discovery_service_uses_settings_weights():
    """DISCOVERY_NEWS_WEIGHTS from settings is picked up."""
    from config.config import Settings
    from services.discovery_service import DiscoveryService

    s = Settings(NICEGUI_STORAGE_SECRET="x")
    svc = DiscoveryService(settings=s)
    assert svc.settings.DISCOVERY_NEWS_WEIGHTS["recency"] == 0.50
