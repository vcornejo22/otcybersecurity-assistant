"""Retriever factory for the industrial cybersecurity (OT/ICS) RAG pipeline."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from app.config import Settings
from app.rag.embeddings import OpenAICompatibleEmbeddings
from app.rag.llm import OpenAICompatibleChatModel
from app.rag.prompts import MULTI_QUERY_TEMPLATE


def _get_chroma_client(settings: Settings) -> chromadb.HttpClient:
    """Return a ChromaDB HTTP client connected to the server."""
    return chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def load_vectorstore(settings: Settings | None = None) -> Chroma:
    """Load the configured ChromaDB collection from the HTTP server."""
    settings = settings or Settings()
    embeddings = OpenAICompatibleEmbeddings(settings=settings)
    client = _get_chroma_client(settings)
    return Chroma(
        client=client,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
    )


def get_chroma_client(settings: Settings | None = None) -> chromadb.HttpClient:
    """Return the ChromaDB HTTP client (for health checks, etc.)."""
    settings = settings or Settings()
    return _get_chroma_client(settings)


def build_retriever(
    settings: Settings | None = None,
    enable_multi_query_override: bool | None = None,
):
    """Build the configured retrieval pipeline.

    The pipeline consists of:
      - MMR retriever, optionally wrapped with MultiQueryRetriever for
        LLM-based query expansion when ``ENABLE_MULTI_QUERY`` is enabled
        (disabled by default to avoid an extra remote LLM call).
      - Optional EnsembleRetriever combining MMR and similarity search.

    Args:
        settings: Application settings.
        enable_multi_query_override: Per-request override for the
            ``ENABLE_MULTI_QUERY`` setting.  ``None`` = use the setting.
    """
    settings = settings or Settings()
    vectorstore = load_vectorstore(settings=settings)

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.TOP_K_DEFAULT,
            "lambda_mult": settings.MMR_LAMBDA_MULT,
            "fetch_k": settings.MMR_FETCH_K,
        },
    )

    base_retriever = mmr_retriever
    effective_mq = (
        settings.ENABLE_MULTI_QUERY
        if enable_multi_query_override is None
        else enable_multi_query_override
    )
    if effective_mq:
        llm = OpenAICompatibleChatModel(settings=settings, temperature=0.0, max_tokens=200)
        base_retriever = MultiQueryRetriever.from_llm(
            retriever=mmr_retriever,
            llm=llm,
            prompt=MULTI_QUERY_TEMPLATE,
        )

    if not settings.ENABLE_HYBRID_SEARCH:
        return base_retriever

    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": settings.TOP_K_DEFAULT,
            "score_threshold": settings.SIMILARITY_THRESHOLD,
        },
    )

    return EnsembleRetriever(
        retrievers=[base_retriever, similarity_retriever],
        weights=[0.7, 0.3],
    )
