import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.middleware.auth import CurrentApiKey
from app.models.api_key import ApiKey
from app.services.cache import SemanticCache
from app.services.rate_limiter import RateLimiter
from app.services.router import provider_router
from app.services.usage_tracker import track_usage

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[list[str] | str] = None
    stream: bool = False
    # Extra fields forwarded as-is
    user: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    api_key: CurrentApiKey,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    Drop-in replacement: set base_url=http://your-gateway/
    """
    import time

    messages = [m.model_dump() for m in body.messages]

    # --- Rate limiting ---
    rate_limiter: RateLimiter = request.app.state.rate_limiter
    rl_result = await rate_limiter.check_and_increment(
        api_key_id=api_key.id,
        rpm_limit=api_key.rpm_limit,
        tpd_limit=api_key.tpd_limit,
        tokens_used=0,  # Pre-request check; actual tokens logged after
    )
    if not rl_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {rl_result.reason}",
            headers={
                "X-RateLimit-Remaining-RPM": str(rl_result.remaining_rpm),
                "X-RateLimit-Remaining-TPD": str(rl_result.remaining_tpd),
                "Retry-After": "60",
            },
        )

    # --- Cache check ---
    cache: SemanticCache = request.app.state.cache
    cache_hit = False
    cached_response = await cache.get(messages, body.model)

    if cached_response:
        cache_hit = True
        provider_response = cached_response
    else:
        # --- Route to provider ---
        kwargs: dict[str, Any] = {}
        if body.temperature is not None:
            kwargs["temperature"] = body.temperature
        if body.max_tokens is not None:
            kwargs["max_tokens"] = body.max_tokens
        if body.top_p is not None:
            kwargs["top_p"] = body.top_p
        if body.stop is not None:
            kwargs["stop"] = body.stop

        try:
            provider_response, provider_used = await provider_router.route(
                messages, body.model, **kwargs
            )
        except Exception as exc:
            # Log failed request
            await track_usage(
                db=db,
                api_key_id=api_key.id,
                org_id=api_key.org_id,
                provider="unknown",
                model=body.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                status="error",
            )
            logger.error("All providers failed for model=%s: %s", body.model, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"All upstream providers failed: {exc}",
            )

        # Store in cache (async, non-blocking for response)
        await cache.store(messages, body.model, provider_response)

    # --- Log usage ---
    await track_usage(
        db=db,
        api_key_id=api_key.id,
        org_id=api_key.org_id,
        provider=provider_response.provider,
        model=provider_response.model,
        prompt_tokens=provider_response.prompt_tokens,
        completion_tokens=provider_response.completion_tokens,
        latency_ms=provider_response.latency_ms,
        status="success",
    )

    # Record tokens for TPD tracking
    await rate_limiter.record_tokens(
        api_key_id=api_key.id,
        tokens=provider_response.prompt_tokens + provider_response.completion_tokens,
    )

    # --- Build OpenAI-format response ---
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    response = ChatCompletionResponse(
        id=response_id,
        created=created,
        model=provider_response.model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=provider_response.content),
                finish_reason=provider_response.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=provider_response.prompt_tokens,
            completion_tokens=provider_response.completion_tokens,
            total_tokens=provider_response.prompt_tokens + provider_response.completion_tokens,
        ),
    )

    from fastapi.responses import JSONResponse

    headers = {
        "X-Cache-Hit": str(cache_hit).lower(),
        "X-Provider": provider_response.provider,
        "X-Latency-Ms": str(provider_response.latency_ms),
    }

    return JSONResponse(content=response.model_dump(), headers=headers)
