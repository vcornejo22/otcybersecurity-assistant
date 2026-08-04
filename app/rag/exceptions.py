"""Domain exceptions for the RAG pipeline."""


class RAGError(Exception):
    """Base class for RAG pipeline errors."""


class LLMUnavailableError(RAGError):
    """Raised when the Nan Builders LLM API is unreachable or returns an error."""


class IngestionError(RAGError):
    """Raised when document ingestion fails."""
