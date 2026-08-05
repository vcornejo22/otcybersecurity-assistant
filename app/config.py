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
    APP_NAME: str = os.getenv("APP_NAME", "otcybersecurity-assistant")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    APP_CORS_ORIGINS: str = os.getenv(
        "APP_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
    )

    # LLM API (OpenAI-compatible: OpenAI, Qwen, etc.)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")

    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    API_KEY: str = os.getenv("API_KEY", "")
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "30/minute")

    # Model configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.7-max")
    LLM_MODEL_EMBEDDING: str = os.getenv("LLM_MODEL_EMBEDDING", "qwen3-embedding")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    # Qwen3 reasoning: false = responde directo (más rápido, content nunca vacío)
    LLM_ENABLE_THINKING: bool = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"

    # ChromaDB configuration (HTTP server mode)
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "iec62443_docs")

    # Retrieval configuration
    TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    MMR_FETCH_K: int = 20
    MMR_LAMBDA_MULT: float = 0.7
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "false").lower() == "true"
    ENABLE_QUERY_TRANSLATION: bool = (
        os.getenv("ENABLE_QUERY_TRANSLATION", "true").lower() == "true"
    )
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
