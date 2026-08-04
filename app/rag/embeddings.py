"""Nan Builders embedding wrapper for LangChain."""

import httpx
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.rag.exceptions import RAGError


class NanBuildingsEmbeddings(Embeddings):
    """LangChain-compatible embedding client for the Nan Builders API.

    Calls ``POST /v1/embeddings`` with the configured embedding model
    and returns the embedding vectors. The API key and base URL are read
    from the application settings.

    Respects Nan Builders batch limit of 32 texts per request.
    """

    _BATCH_SIZE = 32

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        if not self.settings.LLM_API_KEY:
            raise RAGError(
                "LLM_API_KEY is not configured. Set it in your environment or .env file."
            )
        self._client = httpx.Client(
            base_url=self.settings.LLM_BASE_URL,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of documents, batched in groups of 32."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            response = self._client.post(
                "/embeddings",
                json={
                    "model": self.settings.LLM_MODEL_EMBEDDING,
                    "input": batch,
                },
            )
            response.raise_for_status()
            payload = response.json()

            try:
                data = payload["data"]
            except KeyError as exc:
                raise RAGError(f"Unexpected embeddings response format: {payload}") from exc

            sorted_data = sorted(data, key=lambda item: item.get("index", 0))
            all_embeddings.extend([item["embedding"] for item in sorted_data])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding for a single query string."""
        response = self._client.post(
            "/embeddings",
            json={
                "model": self.settings.LLM_MODEL_EMBEDDING,
                "input": text,
            },
        )
        response.raise_for_status()
        payload = response.json()

        try:
            return payload["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise RAGError(f"Unexpected embeddings response format: {payload}") from exc
