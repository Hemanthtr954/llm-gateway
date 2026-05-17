from datetime import datetime
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

router = APIRouter()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "llm_gateway_requests_total",
    "Total number of chat completion requests",
    ["provider", "model", "status"],
)

REQUEST_LATENCY = Histogram(
    "llm_gateway_request_latency_ms",
    "Request latency in milliseconds",
    ["provider", "model"],
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000, 30000],
)

CACHE_HITS = Counter(
    "llm_gateway_cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISSES = Counter(
    "llm_gateway_cache_misses_total",
    "Total number of cache misses",
)

TOKENS_USED = Counter(
    "llm_gateway_tokens_total",
    "Total tokens processed",
    ["provider", "model", "type"],  # type: prompt | completion
)

COST_USD = Counter(
    "llm_gateway_cost_usd_total",
    "Total cost in USD",
    ["provider", "model"],
)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Any:
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
