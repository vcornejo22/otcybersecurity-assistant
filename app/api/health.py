"""Health check endpoint."""

import time
from typing import Any

import chromadb
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings

router = APIRouter()

# Recorded by app lifespan on startup.
_start_time: float | None = None


def _rag_status(settings: Settings) -> dict[str, Any]:
    """Return RAG health status by inspecting the configured ChromaDB collection."""
    try:
        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
        collection = client.get_collection(name=settings.CHROMA_COLLECTION)
        chunks_count = collection.count()

        if chunks_count == 0:
            return {
                "rag_loaded": False,
                "chroma_collection": settings.CHROMA_COLLECTION,
                "documents_count": 0,
                "chunks_count": 0,
            }

        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        filenames = {meta.get("filename") for meta in metadatas if meta and meta.get("filename")}

        return {
            "rag_loaded": True,
            "chroma_collection": settings.CHROMA_COLLECTION,
            "documents_count": len(filenames),
            "chunks_count": chunks_count,
        }
    except Exception:
        return {
            "rag_loaded": False,
            "chroma_collection": settings.CHROMA_COLLECTION,
            "documents_count": 0,
            "chunks_count": 0,
        }


@router.get("/api/health")
async def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return service health status, uptime, and RAG readiness."""
    rag = _rag_status(settings)
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "uptime_seconds": round(time.monotonic() - _start_time, 2)
        if _start_time is not None
        else 0.0,
        **rag,
    }
