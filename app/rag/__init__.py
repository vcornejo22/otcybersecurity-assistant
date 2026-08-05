"""RAG pipeline package (H1)."""

from app.rag.embeddings import OpenAICompatibleEmbeddings
from app.rag.exceptions import IngestionError, LLMUnavailableError, RAGError
from app.rag.generator import GenerationResult, generate_answer
from app.rag.llm import OpenAICompatibleChatModel
from app.rag.retrieval import build_retriever

__all__ = [
    "OpenAICompatibleEmbeddings",
    "OpenAICompatibleChatModel",
    "build_retriever",
    "generate_answer",
    "GenerationResult",
    "RAGError",
    "LLMUnavailableError",
    "IngestionError",
]
