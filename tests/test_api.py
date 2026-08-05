"""API tests for the RAG query and health endpoints."""

from unittest.mock import Mock, patch

from app.config import get_settings
from app.rag.exceptions import LLMUnavailableError
from app.rag.generator import GenerationResult


class TestQueryEndpoint:
    """Tests for POST /api/query."""

    def test_valid_question_returns_query_response(self, test_client, auth_headers):
        """A valid question returns a 200 QueryResponse with sources."""
        result = GenerationResult(
            answer="IEC 62443 is a series of standards.",
            sources=[
                {
                    "fragment": 1,
                    "filename": "iec62443.pdf",
                    "page_number": 1,
                    "excerpt": "IEC 62443 is a series...",
                    "score": 0.95,
                }
            ],
            model="qwen3.7-max",
            tokens_used=42,
            latency_ms=123.0,
        )

        with (
            patch("app.api.routes.build_retriever") as mock_build,
            patch("app.api.routes.generate_answer") as mock_generate,
        ):
            mock_build.return_value = Mock()
            mock_generate.return_value = result
            response = test_client.post(
                "/api/query",
                json={"question": "What is IEC 62443?"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "IEC 62443 is a series of standards."
        assert body["model"] == "qwen3.7-max"
        assert body["tokens_used"] == 42
        assert "latency_ms" in body
        assert len(body["sources"]) == 1
        assert body["sources"][0]["filename"] == "iec62443.pdf"

    def test_empty_question_returns_422(self, test_client, auth_headers):
        """A question shorter than 5 characters returns a 422 validation error."""
        response = test_client.post("/api/query", json={"question": "abc"}, headers=auth_headers)
        assert response.status_code == 422

    def test_long_question_returns_422(self, test_client, auth_headers):
        """A question longer than 500 characters returns a 422 validation error."""
        response = test_client.post(
            "/api/query", json={"question": "x" * 501}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_invalid_top_k_returns_422(self, test_client, auth_headers):
        """top_k outside the allowed range returns a 422 validation error."""
        response = test_client.post(
            "/api/query",
            json={"question": "What is IEC 62443?", "top_k": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_temperature_returns_422(self, test_client, auth_headers):
        """temperature outside the allowed range returns a 422 validation error."""
        response = test_client.post(
            "/api/query",
            json={"question": "What is IEC 62443?", "temperature": 1.5},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_llm_unavailable_returns_503(self, test_client, auth_headers):
        """When the LLM raises LLMUnavailableError, the endpoint returns 503."""
        with (
            patch("app.api.routes.build_retriever") as mock_build,
            patch("app.api.routes.generate_answer") as mock_generate,
        ):
            mock_build.return_value = Mock()
            mock_generate.side_effect = LLMUnavailableError("API unreachable")

            response = test_client.post(
                "/api/query",
                json={"question": "What is IEC 62443?"},
                headers=auth_headers,
            )

        assert response.status_code == 503

    def test_query_handles_empty_collection(self, test_client, auth_headers):
        """When the retriever returns no documents, the endpoint returns a graceful answer."""
        result = GenerationResult(
            answer="No relevant documents found.",
            sources=[],
            model="qwen3.7-max",
            tokens_used=0,
            latency_ms=0.0,
        )

        with (
            patch("app.api.routes.build_retriever") as mock_build,
            patch("app.api.routes.generate_answer") as mock_generate,
        ):
            mock_build.return_value = Mock()
            mock_generate.return_value = result
            response = test_client.post(
                "/api/query",
                json={"question": "What is IEC 62443?"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert "No relevant documents found" in body["answer"]
        assert body["sources"] == []

    def test_query_without_auth_returns_401(self, test_client):
        """A query without Authorization header returns 401."""
        response = test_client.post("/api/query", json={"question": "What is IEC 62443?"})
        assert response.status_code == 401


class TestLoginEndpoint:
    """Tests for POST /api/auth/login."""

    def test_login_is_rate_limited(self, test_client):
        """Exceeding RATE_LIMIT on login returns 429 (Too Many Requests)."""
        limit = int(get_settings().RATE_LIMIT.split("/")[0])
        response = None
        for _ in range(limit + 1):
            response = test_client.post(
                "/api/auth/login", json={"user": "t", "password": "wrong-key"}
            )
        assert response is not None
        assert response.status_code == 429


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_rag_status(self, test_client, mock_chroma):
        """The health endpoint returns RAG counts when ChromaDB is available."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "uptime_seconds" in body
        assert body["rag_loaded"] is True
        assert body["chroma_collection"] == "iec62443_docs"
        assert body["documents_count"] == 1
        assert body["chunks_count"] == 2

    def test_health_handles_missing_collection(self, test_client, monkeypatch):
        """The health endpoint returns zeros when the collection is missing."""

        def raise_not_found(*args, **kwargs):
            raise Exception("collection not found")

        monkeypatch.setattr("app.api.health.chromadb.HttpClient", raise_not_found)
        response = test_client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["rag_loaded"] is False
        assert body["documents_count"] == 0
        assert body["chunks_count"] == 0
