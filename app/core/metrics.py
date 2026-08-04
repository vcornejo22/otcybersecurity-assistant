"""Prometheus metrics for the IEC 62443 Assistant."""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest

QUERY_COUNT = Counter(
    "otcybersecurity_queries_total",
    "Total de consultas al asistente",
    ["status"],
)

QUERY_LATENCY = Histogram(
    "otcybersecurity_query_latency_seconds",
    "Latencia de consultas en segundos",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

CHROMA_CHUNKS = Gauge(
    "otcybersecurity_chroma_chunks_total",
    "Total de chunks en ChromaDB",
)

LLM_TOKENS = Counter(
    "otcybersecurity_llm_tokens_total",
    "Total de tokens consumidos por el LLM",
)


def setup_metrics(app: FastAPI) -> None:
    """Register metrics endpoint and middleware on the FastAPI app."""

    @app.get("/api/metrics", include_in_schema=False)
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type="text/plain")
