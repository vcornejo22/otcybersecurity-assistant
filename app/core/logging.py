"""Structured logging with request ID tracing."""

import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from loguru import logger

request_id_headers: dict[str, str] = {}


def setup_logging(app: FastAPI) -> None:
    """Register structured logging middleware on the FastAPI app."""

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id

        logger.info(f"[{request_id}] {request.method} {request.url.path} — started")
        start = time.monotonic()

        response = await call_next(request)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"— {response.status_code} ({elapsed:.0f}ms)"
        )

        response.headers["X-Request-ID"] = request_id
        return response
