"""Answer generation for the industrial cybersecurity (OT/ICS) RAG pipeline."""

import time
from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import Settings
from app.rag.exceptions import LLMUnavailableError
from app.rag.llm import NanBuildersChatModel
from app.rag.prompts import RAG_PROMPT, TRANSLATE_QUERY_TEMPLATE


@dataclass
class GenerationResult:
    """Result of a RAG generation call."""

    answer: str
    sources: list[dict]
    model: str
    tokens_used: int
    latency_ms: float


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into the prompt context."""
    formatted: list[str] = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page_number", "unknown")
        header = f"[Fragment {i}] - Source: {filename} - Page: {page}"
        content = doc.page_content.strip()
        formatted.append(f"{header}\n{content}")
    return "\n\n".join(formatted)


def _build_sources(docs: list[Document]) -> list[dict]:
    """Build serializable source metadata from retrieved documents."""
    sources = []
    for i, doc in enumerate(docs, start=1):
        sources.append(
            {
                "fragment": i,
                "filename": doc.metadata.get("filename", "unknown"),
                "page_number": doc.metadata.get("page_number", "unknown"),
                "excerpt": doc.page_content.strip(),
                "score": doc.metadata.get("score"),
            }
        )
    return sources


def generate_answer(
    question: str,
    retriever,
    settings: Settings | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
    enable_query_translation: bool | None = None,
) -> GenerationResult:
    """Retrieve relevant documents and generate a cited answer.

    Args:
        question: The user's question.
        retriever: A LangChain retriever (e.g. from ``build_retriever``).
        settings: Optional settings override.
        temperature: Optional generation temperature override.
        enable_thinking: Per-request override for ``LLM_ENABLE_THINKING``.
            ``None`` = use the setting.
        enable_query_translation: Per-request override for
            ``ENABLE_QUERY_TRANSLATION``.  When enabled, the question is
            translated to English before retrieval (documents are in both
            languages), and the original question is used for generation.

    Returns:
        A ``GenerationResult`` containing the answer, sources, and metadata.

    Raises:
        LLMUnavailableError: If the LLM call fails or returns a malformed response.
    """
    settings = settings or Settings()

    # ── Optional: translate query to English for retrieval ──────
    retriever_question = question
    effective_translation = (
        settings.ENABLE_QUERY_TRANSLATION
        if enable_query_translation is None
        else enable_query_translation
    )
    if effective_translation:
        trans_llm = NanBuildersChatModel(
            settings=settings, max_tokens=200, temperature=0.0
        )
        trans_prompt = TRANSLATE_QUERY_TEMPLATE.format(question=question)
        trans_response = trans_llm.invoke(trans_prompt)
        translated = str(trans_response.content).strip()
        if translated:
            retriever_question = translated

    docs = retriever.invoke(retriever_question)

    if not docs:
        return GenerationResult(
            answer=(
                "No se encontraron documentos relevantes de ciberseguridad industrial "
                "para responder esta pregunta. "
                "Intentá reformular la consulta o ingerí documentos adicionales."
            ),
            sources=[],
            model=settings.LLM_MODEL,
            tokens_used=0,
            latency_ms=0.0,
        )

    context = _format_context(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)

    llm_kwargs = {"settings": settings}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if enable_thinking is not None:
        llm_kwargs["enable_thinking"] = enable_thinking
    llm = NanBuildersChatModel(**llm_kwargs)

    start = time.perf_counter()
    try:
        response = llm.invoke(prompt)
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(
            f"Unexpected error while calling Nan Builders LLM: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    answer = str(response.content)
    tokens_used = response.response_metadata.get("tokens_used", 0)

    sources = _build_sources(docs)

    return GenerationResult(
        answer=answer,
        sources=sources,
        model=settings.LLM_MODEL,
        tokens_used=tokens_used,
        latency_ms=round(latency_ms, 2),
    )
