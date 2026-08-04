"""Application configuration via pydantic-settings."""

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carga explícita del .env (en Docker no existe el archivo
# pero las variables ya están en el entorno vía docker-compose env_file)
load_dotenv(override=True, verbose=False)


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env file."""

    # Application metadata
    APP_NAME: str = os.getenv("APP_NAME")
    APP_VERSION: str = os.getenv("APP_VERSION")
    APP_HOST: str = os.getenv("APP_HOST")
    APP_PORT: int = int(os.getenv("APP_PORT"))
    APP_CORS_ORIGINS: str = os.getenv("APP_CORS_ORIGINS")

    # Nan Builders API
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL")

    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    API_KEY: str = os.getenv("API_KEY", "")
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "30/minute")

    # Model configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL")
    LLM_MODEL_EMBEDDING: str = os.getenv("LLM_MODEL_EMBEDDING")

    # ChromaDB configuration (HTTP server mode)
    CHROMA_HOST: str = os.getenv("CHROMA_HOST")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT"))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION")

    # Retrieval configuration
    TOP_K_DEFAULT: int = 3
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))
    MMR_FETCH_K: int = 20
    MMR_LAMBDA_MULT: float = 0.7
    ENABLE_HYBRID_SEARCH: bool = True
    SIMILARITY_THRESHOLD: float = 0.70
    RETRIEVER_SEARCH_TYPE: str = "mmr"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Return CORS origins as a list from the comma-separated string."""
        return [origin.strip() for origin in self.APP_CORS_ORIGINS.split(",") if origin.strip()]


# Singleton instance for dependency injection.
_settings = Settings()


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return _settings
