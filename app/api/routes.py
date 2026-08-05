"""RAG query API router with auth and rate limiting."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.deps import verify_credentials
from app.api.schemas import QueryRequest, QueryResponse, Source
from app.config import Settings, get_settings
from app.core.auth import create_jwt, verify_api_key
from app.core.metrics import LLM_TOKENS, QUERY_COUNT, QUERY_LATENCY
from app.core.rate_limit import limiter
from app.rag.exceptions import LLMUnavailableError
from app.rag.generator import generate_answer
from app.rag.retrieval import build_retriever

router = APIRouter()


class LoginRequest(BaseModel):
    user: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/api/auth/login", response_model=LoginResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT)
async def login(
    request: Request,
    body: LoginRequest,
    settings: Settings = Depends(get_settings),
) -> LoginResponse:
    """Authenticate and return a JWT token."""
    if not verify_api_key(body.password, settings):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = create_jwt(body.user, settings=settings)
    return LoginResponse(access_token=token)


@router.post("/api/query", response_model=QueryResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT)
async def query(
    request: Request,
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    user: str = Depends(verify_credentials),
) -> QueryResponse:
    """Answer an IEC 62443 question using the RAG pipeline."""
    start = time.perf_counter()
    try:
        retriever = build_retriever(settings, enable_multi_query_override=body.enable_multi_query)
        result = generate_answer(
            body.question,
            retriever,
            settings,
            temperature=body.temperature,
            enable_thinking=body.enable_thinking,
            enable_query_translation=body.enable_query_translation,
        )
    except LLMUnavailableError:
        QUERY_COUNT.labels(status="503").inc()
        raise
    except Exception as exc:
        QUERY_COUNT.labels(status="500").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado al procesar la consulta: {exc}",
        ) from exc

    latency_s = time.perf_counter() - start
    QUERY_LATENCY.observe(latency_s)
    QUERY_COUNT.labels(status="200").inc()
    LLM_TOKENS.inc(result.tokens_used)

    latency_ms = latency_s * 1000
    sources = [Source(**source) for source in result.sources]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        model=result.model,
        tokens_used=result.tokens_used,
        latency_ms=round(latency_ms, 2),
    )
