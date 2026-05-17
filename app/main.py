import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routers import admin, chat, health, keys, models
from app.services.cache import SemanticCache
from app.services.rate_limiter import RateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    logger.info("Starting LLM Gateway...")

    # Initialize database (create tables)
    await init_db()

    # Initialize Redis client
    redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    try:
        await redis_client.ping()
        logger.info("Redis connected at %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — rate limiting will fail-open", exc)

    # Attach shared services to app state
    app.state.redis = redis_client
    app.state.rate_limiter = RateLimiter(redis_client)
    app.state.cache = SemanticCache(redis_client)

    logger.info("LLM Gateway ready.")
    yield

    # Cleanup
    await redis_client.aclose()
    logger.info("LLM Gateway shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "OpenAI-compatible LLM Gateway with multi-provider routing, "
            "per-team cost tracking, semantic caching, and automatic fallback."
        ),
        lifespan=lifespan,
    )

    # CORS — allow all origins for gateway use-cases
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, tags=["Observability"])
    app.include_router(models.router, tags=["Models"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(keys.router, tags=["API Keys"])
    app.include_router(admin.router, tags=["Admin"])

    return app


app = create_app()
