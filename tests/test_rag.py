"""Unit tests for the RAG pipeline components."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.rag.embeddings import NanBuildingsEmbeddings
from app.rag.exceptions import LLMUnavailableError
from app.rag.generator import GenerationResult, generate_answer
from app.rag.llm import NanBuildersChatModel
from app.rag.retrieval import build_retriever


class TestNanBuildingsEmbeddings:
    """Tests for the Nan Builders embedding wrapper."""

    def test_embed_query_returns_1024_dim_vector(self):
        """A successful embeddings call returns a 1024-dimension vector."""
        fake_response = {
            "data": [{"index": 0, "embedding": [0.1] * 1024}],
            "model": "qwen3-embedding",
        }

        with patch("httpx.Client.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = fake_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            embeddings = NanBuildingsEmbeddings(
                settings=Mock(
                    LLM_API_KEY="test",
                    LLM_BASE_URL="http://test",
                    LLM_MODEL_EMBEDDING="qwen3-embedding",
                )
            )
            vector = embeddings.embed_query("What is IEC 62443?")

        assert len(vector) == 1024
        assert vector[0] == pytest.approx(0.1)

    def test_embed_query_raises_on_missing_key(self):
        """The wrapper raises a clear error when the API key is missing."""
        from app.rag.exceptions import RAGError

        with pytest.raises(RAGError):
            NanBuildingsEmbeddings(settings=Mock(LLM_API_KEY="", LLM_BASE_URL="http://test"))


class TestBuildRetriever:
    """Tests for the retrieval factory."""

    def test_build_retriever_skips_multi_query_by_default(self, test_settings, monkeypatch):
        """Multi-query is off by default, so no LLM query-expansion call is made."""
        monkeypatch.setattr(
            "app.rag.retrieval._get_chroma_client",
            lambda settings: MagicMock(),
        )
        monkeypatch.setattr(
            "app.rag.retrieval.Chroma",
            lambda *args, **kwargs: MagicMock(
                as_retriever=lambda **kwargs: MagicMock(spec="Retriever")
            ),
        )
        from_llm = MagicMock(return_value=MagicMock(spec="Retriever"))
        monkeypatch.setattr(
            "app.rag.retrieval.MultiQueryRetriever.from_llm",
            from_llm,
        )
        monkeypatch.setattr(
            "app.rag.retrieval.EnsembleRetriever",
            lambda *args, **kwargs: MagicMock(spec="Retriever"),
        )

        retriever = build_retriever(test_settings)

        assert retriever is not None
        from_llm.assert_not_called()

    def test_build_retriever_with_multi_query_enabled(self, test_settings, monkeypatch):
        """build_retriever wraps MMR with MultiQueryRetriever when ENABLE_MULTI_QUERY is on."""
        test_settings.ENABLE_MULTI_QUERY = True
        monkeypatch.setattr(
            "app.rag.retrieval._get_chroma_client",
            lambda settings: MagicMock(),
        )
        monkeypatch.setattr(
            "app.rag.retrieval.Chroma",
            lambda *args, **kwargs: MagicMock(
                as_retriever=lambda **kwargs: MagicMock(spec="Retriever")
            ),
        )
        from_llm = MagicMock(return_value=MagicMock(spec="Retriever"))
        monkeypatch.setattr(
            "app.rag.retrieval.MultiQueryRetriever.from_llm",
            from_llm,
        )
        monkeypatch.setattr(
            "app.rag.retrieval.EnsembleRetriever",
            lambda *args, **kwargs: MagicMock(spec="Retriever"),
        )

        retriever = build_retriever(test_settings)

        assert retriever is not None
        from_llm.assert_called_once()


class TestGenerateAnswer:
    """Tests for the answer generator."""

    def test_generate_answer_with_mock_retriever(self, test_settings):
        """A successful generation returns a structured result."""
        retriever = Mock()
        retriever.invoke.return_value = [
            Mock(
                page_content="IEC 62443 is a series of standards.",
                metadata={"filename": "iec62443.pdf", "page_number": 1},
            )
        ]

        llm_response = Mock()
        llm_response.content = "It is a series of cybersecurity standards."
        llm_response.response_metadata = {"tokens_used": 42}

        with patch("app.rag.generator.NanBuildersChatModel") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = llm_response
            result = generate_answer(
                "What is IEC 62443?",
                retriever,
                test_settings,
            )

        assert isinstance(result, GenerationResult)
        assert result.answer == "It is a series of cybersecurity standards."
        assert result.tokens_used == 42
        assert len(result.sources) == 1
        assert result.sources[0]["filename"] == "iec62443.pdf"

    def test_generate_answer_handles_empty_retriever(self, test_settings):
        """When no documents are retrieved, the generator returns a graceful message."""
        retriever = Mock()
        retriever.invoke.return_value = []

        result = generate_answer(
            "What is IEC 62443?",
            retriever,
            test_settings,
        )

        assert "No se encontraron documentos relevantes" in result.answer
        assert result.sources == []
        assert result.tokens_used == 0

    def test_generate_answer_raises_llm_unavailable(self, test_settings):
        """An LLM failure raises LLMUnavailableError."""
        retriever = Mock()
        retriever.invoke.return_value = [
            Mock(
                page_content="Some content",
                metadata={"filename": "doc.pdf", "page_number": 1},
            )
        ]

        with patch("app.rag.generator.NanBuildersChatModel") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.side_effect = LLMUnavailableError("API down")
            with pytest.raises(LLMUnavailableError):
                generate_answer(
                    "What is IEC 62443?",
                    retriever,
                    test_settings,
                )


class TestNanBuildersChatModel:
    """Tests for the LLM wrapper."""

    def test_raises_on_missing_api_key(self):
        """The LLM wrapper requires a configured API key."""
        from app.config import Settings

        with pytest.raises(LLMUnavailableError):
            NanBuildersChatModel(settings=Settings(LLM_API_KEY="", LLM_BASE_URL="http://test"))
