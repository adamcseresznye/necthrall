import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_config_module_from_path(env: dict):
    """Load config/config.py as an isolated module with given env mapping.

    We load the file directly to avoid earlier-import interference from other
    parts of the codebase or pytest collection.
    """
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.py"
    assert cfg_path.exists(), f"Config file not found at {cfg_path}"
    # Ensure the repo .env is not loaded during tests
    env = {**env, "SKIP_DOTENV_LOADER": "1"}

    with patch.dict(os.environ, env, clear=True):
        spec = importlib.util.spec_from_file_location("_temp_config", str(cfg_path))
        module = importlib.util.module_from_spec(spec)
        # Execute module in isolated namespace
        spec.loader.exec_module(module)
        return module


@pytest.mark.unit
def test_config_missing_semantic_scholar_key():
    """Missing Semantic Scholar key should raise ValueError with help URL."""
    env = {
        "GOOGLE_API_KEY": "fake_google",
        "GROQ_API_KEY": "fake_groq",
    }
    with pytest.raises(ValueError) as exc_info:
        _load_config_module_from_path(env)

    msg = str(exc_info.value)
    assert "SEMANTIC_SCHOLAR_API_KEY" in msg
    assert "semanticscholar.org" in msg


@pytest.mark.unit
def test_config_missing_google_key():
    """Missing GOOGLE_API_KEY should raise ValueError with help URL."""
    env = {
        "SEMANTIC_SCHOLAR_API_KEY": "fake_ss",
        "GROQ_API_KEY": "fake_groq",
    }
    with pytest.raises(ValueError) as exc_info:
        _load_config_module_from_path(env)

    msg = str(exc_info.value)
    assert "GOOGLE_API_KEY" in msg or "ai.google.dev" in msg


@pytest.mark.unit
def test_config_missing_groq_key():
    """Missing GROQ_API_KEY should raise ValueError with help URL."""
    env = {
        "SEMANTIC_SCHOLAR_API_KEY": "fake_ss",
        "GOOGLE_API_KEY": "fake_google",
    }
    with pytest.raises(ValueError) as exc_info:
        _load_config_module_from_path(env)

    msg = str(exc_info.value)
    assert "GROQ_API_KEY" in msg or "console.groq.com" in msg


@pytest.mark.unit
def test_config_all_keys_present():
    """Config validation passes when all keys are present."""
    env = {
        "SEMANTIC_SCHOLAR_API_KEY": "fake_ss_key",
        "GOOGLE_API_KEY": "fake_google_key",
        "GROQ_API_KEY": "fake_groq_key",
    }
    config = _load_config_module_from_path(env)
    # validate_config called on import; calling again should be OK
    config.validate_config()

    assert config.SEMANTIC_SCHOLAR_API_KEY == "fake_ss_key"
    assert config.GOOGLE_API_KEY == "fake_google_key"
    assert config.GROQ_API_KEY == "fake_groq_key"


@pytest.mark.unit
def test_custom_model_overrides():
    """Custom model names from env vars override the defaults."""
    env = {
        "SEMANTIC_SCHOLAR_API_KEY": "fake_ss_key",
        "GOOGLE_API_KEY": "fake_google_key",
        "GROQ_API_KEY": "fake_groq_key",
        "QUERY_OPTIMIZATION_MODEL": "custom/query-model",
        "SYNTHESIS_MODEL": "custom/synthesis-model",
    }
    config = _load_config_module_from_path(env)
    assert config.QUERY_OPTIMIZATION_MODEL == "custom/query-model"
    assert config.SYNTHESIS_MODEL == "custom/synthesis-model"
