import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # API Keys
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    PRIMARY_LLM_API_KEY: Optional[str] = None
    SECONDARY_LLM_API_KEY: Optional[str] = None
    WEB3FORMS_ACCESS_KEY: Optional[str] = None

    # Models
    QUERY_OPTIMIZATION_MODEL: str
    QUERY_OPTIMIZATION_FALLBACK: str
    SYNTHESIS_MODEL: str
    SYNTHESIS_FALLBACK: str

    NICEGUI_STORAGE_SECRET: str

    # Tuning
    RAG_RETRIEVAL_TOP_K: int = 50
    RAG_PASSAGES_TOP_K: int = 12
    TIMEOUT: int = 30

    # Research loop
    RESEARCH_MAX_ROUNDS: int = 3
    RESEARCH_PDF_TARGETS: dict[int, int] = Field(
        default_factory=lambda: {1: 3, 2: 2, 3: 1}
    )

    # Acquisition
    ACQUISITION_PER_PDF_TIMEOUT: float = 30.0
    ACQUISITION_CHUNK_SIZE: int = 32768  # 32 * 1024
    ACQUISITION_TARGET_PDF_COUNT: int = 5

    # Discovery scoring weights
    DISCOVERY_DEFAULT_WEIGHTS: dict[str, float] = Field(
        default_factory=lambda: {"relevance": 0.60, "authority": 0.35, "recency": 0.05}
    )
    DISCOVERY_NEWS_WEIGHTS: dict[str, float] = Field(
        default_factory=lambda: {"relevance": 0.50, "authority": 0.0, "recency": 0.50}
    )
    DISCOVERY_FOUNDATIONAL_WEIGHTS: dict[str, float] = Field(
        default_factory=lambda: {"relevance": 0.40, "authority": 0.60, "recency": 0.0}
    )

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
