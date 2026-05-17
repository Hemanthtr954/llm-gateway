import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.usage_log import UsageLog

logger = logging.getLogger(__name__)

# Cost per 1,000 tokens: (prompt_cost_per_1k, completion_cost_per_1k)
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    # Anthropic
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-sonnet-20240620": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    # Groq (very cheap)
    "llama-3.1-8b-instant": (0.00005, 0.0001),
    "llama-3.1-70b-versatile": (0.00059, 0.00079),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "mixtral-8x7b-32768": (0.00024, 0.00024),
    "gemma2-9b-it": (0.0002, 0.0002),
}

# Default pricing for unknown models (use cheapest tier)
DEFAULT_PRICING: tuple[float, float] = (0.001, 0.002)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a request."""
    # Try exact match first, then prefix match
    pricing = PRICING.get(model)
    if pricing is None:
        # Try prefix matching (e.g., "claude-3-5-sonnet" matches "claude-3-5-sonnet-20241022")
        for key, val in PRICING.items():
            if model.startswith(key) or key.startswith(model):
                pricing = val
                break

    if pricing is None:
        pricing = DEFAULT_PRICING

    prompt_cost, completion_cost = pricing
    cost = (prompt_tokens / 1000) * prompt_cost + (completion_tokens / 1000) * completion_cost
    return round(cost, 8)


async def track_usage(
    db: AsyncSession,
    api_key_id: str,
    org_id: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    status: str,
) -> UsageLog:
    """Create and persist a UsageLog record."""
    total_tokens = prompt_tokens + completion_tokens
    cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

    log = UsageLog(
        api_key_id=api_key_id,
        org_id=org_id,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        status=status,
        created_at=datetime.utcnow(),
    )

    db.add(log)
    try:
        await db.flush()  # get the ID assigned without full commit
    except Exception as exc:
        logger.error("Failed to write usage log: %s", exc)
        raise

    return log
