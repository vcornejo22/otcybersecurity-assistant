"""API request/response schemas for the OT/ICS RAG pipeline."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /api/query."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="User question about industrial cybersecurity",
    )
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of relevant fragments to retrieve"
    )
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Generation temperature")
    enable_multi_query: bool | None = Field(
        default=None, description="Override ENABLE_MULTI_QUERY for this request"
    )
    enable_thinking: bool | None = Field(
        default=None, description="Override LLM_ENABLE_THINKING for this request"
    )
    enable_query_translation: bool | None = Field(
        default=None,
        description="Override ENABLE_QUERY_TRANSLATION (traduce la query al "
        "inglés antes de buscar para mejorar recuperación en documentos bilingües)",
    )


class Source(BaseModel):
    """A single source fragment supporting the generated answer."""

    fragment: int = Field(..., description="1-based fragment index")
    filename: str = Field(..., description="Source document filename")
    page_number: int | str = Field(..., description="Page number in the source document")
    excerpt: str = Field(..., description="Chunk excerpt")
    score: float | None = Field(None, description="Relevance score when available")


class QueryResponse(BaseModel):
    """Response body for a successful query."""

    answer: str = Field(..., description="Generated answer")
    sources: list[Source] = Field(default_factory=list, description="Supporting fragments")
    model: str = Field(..., description="LLM model used for generation")
    tokens_used: int = Field(..., description="Total tokens consumed")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response including H0 baseline and RAG status."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    rag_loaded: bool = Field(..., description="Whether the RAG vector store has indexed documents")
    chroma_collection: str = Field(..., description="Configured ChromaDB collection name")
    documents_count: int = Field(..., description="Number of distinct source documents")
    chunks_count: int = Field(..., description="Total number of indexed chunks")
