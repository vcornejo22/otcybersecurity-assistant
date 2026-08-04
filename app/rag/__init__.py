"""RAG pipeline package (H1)."""

from app.rag.embeddings import NanBuildingsEmbeddings
from app.rag.exceptions import IngestionError, LLMUnavailableError, RAGError
from app.rag.generator import GenerationResult, generate_answer
from app.rag.llm import NanBuildersChatModel
from app.rag.retrieval import build_retriever

__all__ = [
    "NanBuildingsEmbeddings",
    "NanBuildersChatModel",
    "build_retriever",
    "generate_answer",
    "GenerationResult",
    "RAGError",
    "LLMUnavailableError",
    "IngestionError",
]
