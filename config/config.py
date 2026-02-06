import logging
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # API Keys
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    PRIMARY_LLM_API_KEY: Optional[str] = None
    SECONDARY_LLM_API_KEY: Optional[str] = None
    WEB3FORMS_ACCESS_KEY: Optional[str] = None

    # Models
    QUERY_OPTIMIZATION_MODEL: str = "mistral/ministral-8b-2512"
    QUERY_OPTIMIZATION_FALLBACK: str = "cerebras/llama3.1-8b"
    SYNTHESIS_MODEL: str = "mistral/mistral-large-2512"
    SYNTHESIS_FALLBACK: str = "cerebras/llama-3.3-70b"

    NICEGUI_STORAGE_SECRET: str

    # Tuning
    RAG_RETRIEVAL_TOP_K: int = 50
    RAG_RERANK_TOP_K: int = 12
    TIMEOUT: int = 30

    # Rate Limiting
    RATE_LIMIT_QUERIES_PER_HOUR: int = 5
    RATE_LIMIT_WINDOW_SECONDS: int = 3600

    # System
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def validate_keys(self):
        """
        Validates that critical API keys are present.
        Logs warnings if they are missing.
        """
        if not self.SEMANTIC_SCHOLAR_API_KEY:
            logger.warning(
                "SEMANTIC_SCHOLAR_API_KEY is not set. Paper discovery may be limited."
            )

        if not self.PRIMARY_LLM_API_KEY:
            logger.warning("PRIMARY_LLM_API_KEY is not set. LLM features will fail.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
