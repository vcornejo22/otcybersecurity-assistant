"""Shared pytest fixtures for the test suite."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


@pytest.fixture(autouse=True)
def _override_settings():
    """Override get_settings dependency so tests use a known API key."""
    test_s = Settings(
        LLM_API_KEY="test-key",
        LLM_BASE_URL="http://test.local",
        JWT_SECRET="test-secret",
        API_KEY="test-api-key",
        CHROMA_HOST="localhost",
        CHROMA_PORT=8001,
        CHROMA_COLLECTION="iec62443_docs",
    )
    app.dependency_overrides[get_settings] = lambda: test_s
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def test_settings() -> Settings:
    """Return settings for direct use in unit tests."""
    return Settings(
        LLM_API_KEY="test-key",
        LLM_BASE_URL="http://test.local",
        JWT_SECRET="test-secret",
        API_KEY="test-api-key",
        CHROMA_HOST="localhost",
        CHROMA_PORT=8001,
        CHROMA_COLLECTION="iec62443_docs",
    )


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authorization headers with a valid test API key."""
    return {"Authorization": "Bearer test-api-key"}


@pytest.fixture
def mock_chroma(monkeypatch) -> MagicMock:
    """Return a mocked ChromaDB collection for health/retrieval tests."""
    collection = MagicMock()
    collection.count.return_value = 2
    collection.get.return_value = {
        "metadatas": [
            {"filename": "doc1.pdf", "page_number": 1},
            {"filename": "doc1.pdf", "page_number": 2},
        ],
    }

    client = MagicMock()
    client.get_collection.return_value = collection

    monkeypatch.setattr("app.api.health.chromadb.HttpClient", lambda *a, **kw: client)
    return collection


@pytest.fixture
def test_client() -> TestClient:
    """Return a configured TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_retriever():
    """Return a retriever that yields two fake documents."""

    def _retriever(query: str):
        return [
            {
                "page_content": "IEC 62443 is a series of standards.",
                "metadata": {
                    "filename": "iec62443.pdf",
                    "page_number": 1,
                    "score": 0.95,
                },
            },
            {
                "page_content": "It covers security for industrial systems.",
                "metadata": {
                    "filename": "iec62443.pdf",
                    "page_number": 2,
                    "score": 0.85,
                },
            },
        ]

    return _retriever
