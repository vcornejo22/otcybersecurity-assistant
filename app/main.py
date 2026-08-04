"""FastAPI application factory."""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import health
from app.api.routes import router as query_router
from app.api.schemas import ErrorResponse
from app.config import get_settings
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics
from app.core.rate_limit import limiter
from app.core.security import setup_security_headers
from app.rag.exceptions import LLMUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Record startup time for uptime calculation."""
    health._start_time = time.monotonic()
    yield


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Rate limit exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security, logging, metrics
setup_security_headers(app)
setup_logging(app)
setup_metrics(app)


# ── Exception handlers ────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a uniform 422 error response for validation failures."""
    messages = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", []))
        messages.append(f"{loc}: {error.get('msg', 'invalid value')}")
    body = ErrorResponse(error={"code": "validation_error", "message": "; ".join(messages)})
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    """Return a 503 response when the Nan Builders LLM is unavailable."""
    body = ErrorResponse(error={"code": "llm_unavailable", "message": str(exc)})
    return JSONResponse(status_code=503, content=body.model_dump())


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 response for unexpected internal failures."""
    body = ErrorResponse(
        error={
            "code": "internal_server_error",
            "message": "Ocurrió un error interno inesperado.",
        }
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# ── Routers ────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query_router)
