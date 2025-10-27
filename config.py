from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Loads and validates application settings from the .env file.
    """
    GEMINI_API_KEY: str = Field(..., alias='GOOGLE_API_KEY')
    GROQ_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        extra='ignore'  # This will ignore extra fields like LLM_MODEL_PRIMARY
    )

settings = Settings()