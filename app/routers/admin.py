import logging
from datetime import datetime, timedelta
from typing import Any, Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.usage_log import UsageLog
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.groq_provider import GroqProvider

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def _verify_master_key(x_master_key: str | None = Header(default=None)) -> None:
    if not x_master_key or x_master_key != settings.master_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Master-Key header.",
        )


MasterKeyDep = Annotated[None, Depends(_verify_master_key)]


class UsageRow(BaseModel):
    org_id: str
    model: str
    day: str
    requests: int
    total_tokens: int
    cost_usd: float


class CostRow(BaseModel):
    org_id: str
    cost_usd: float
    requests: int


class ProviderHealthStatus(BaseModel):
    provider: str
    healthy: bool


class HealthResponse(BaseModel):
    providers: list[ProviderHealthStatus]
    timestamp: str


@router.get("/admin/usage", response_model=list[UsageRow])
async def get_usage(
    _: MasterKeyDep,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Aggregate usage by org / model / day for the past N days."""
    since = datetime.utcnow() - timedelta(days=days)

    result = await db.execute(
        select(
            UsageLog.org_id,
            UsageLog.model,
            func.date(UsageLog.created_at).label("day"),
            func.count(UsageLog.id).label("requests"),
            func.sum(UsageLog.total_tokens).label("total_tokens"),
            func.sum(UsageLog.cost_usd).label("cost_usd"),
        )
        .where(UsageLog.created_at >= since)
        .group_by(UsageLog.org_id, UsageLog.model, func.date(UsageLog.created_at))
        .order_by(func.date(UsageLog.created_at).desc())
    )

    rows = result.all()
    return [
        UsageRow(
            org_id=row.org_id,
            model=row.model,
            day=str(row.day),
            requests=row.requests,
            total_tokens=row.total_tokens or 0,
            cost_usd=round(row.cost_usd or 0.0, 6),
        )
        for row in rows
    ]


@router.get("/admin/costs", response_model=list[CostRow])
async def get_costs(
    _: MasterKeyDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Total cost by org for the current calendar month."""
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    result = await db.execute(
        select(
            UsageLog.org_id,
            func.sum(UsageLog.cost_usd).label("cost_usd"),
            func.count(UsageLog.id).label("requests"),
        )
        .where(
            and_(
                UsageLog.created_at >= month_start,
                UsageLog.status == "success",
            )
        )
        .group_by(UsageLog.org_id)
        .order_by(func.sum(UsageLog.cost_usd).desc())
    )

    rows = result.all()
    return [
        CostRow(
            org_id=row.org_id,
            cost_usd=round(row.cost_usd or 0.0, 4),
            requests=row.requests,
        )
        for row in rows
    ]


@router.get("/admin/health", response_model=HealthResponse)
async def provider_health(_: MasterKeyDep) -> Any:
    """Check connectivity to each upstream provider."""
    import asyncio

    providers = [
        ("openai", OpenAIProvider()),
        ("anthropic", AnthropicProvider()),
        ("groq", GroqProvider()),
    ]

    async def check(name: str, provider: Any) -> ProviderHealthStatus:
        try:
            healthy = await provider.health_check()
        except Exception:
            healthy = False
        return ProviderHealthStatus(provider=name, healthy=healthy)

    results = await asyncio.gather(*[check(n, p) for n, p in providers])

    return HealthResponse(
        providers=list(results),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
