import hashlib
import os
import uuid
import logging
from typing import Any, Annotated

from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.api_key import ApiKey
from app.models.usage_log import UsageLog

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


class CreateKeyRequest(BaseModel):
    org_id: str
    name: str
    rpm_limit: int = 60
    tpd_limit: int = 100_000


class CreateKeyResponse(BaseModel):
    id: str
    key: str  # plaintext — only shown once
    org_id: str
    name: str
    rpm_limit: int
    tpd_limit: int
    message: str = "Store this key securely. It will not be shown again."


class KeySummary(BaseModel):
    id: str
    org_id: str
    name: str
    is_active: bool
    rpm_limit: int
    tpd_limit: int
    created_at: str
    total_requests: int
    total_cost_usd: float


@router.post("/v1/keys", response_model=CreateKeyResponse, status_code=201)
async def create_api_key(
    body: CreateKeyRequest,
    _: MasterKeyDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a new API key. Returns the plaintext key exactly once."""
    # Generate random key
    raw_key = "gw-" + os.urandom(32).hex()
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = str(uuid.uuid4())

    api_key = ApiKey(
        id=key_id,
        key_hash=key_hash,
        org_id=body.org_id,
        name=body.name,
        is_active=True,
        rpm_limit=body.rpm_limit,
        tpd_limit=body.tpd_limit,
    )
    db.add(api_key)
    await db.flush()

    return CreateKeyResponse(
        id=key_id,
        key=raw_key,
        org_id=body.org_id,
        name=body.name,
        rpm_limit=body.rpm_limit,
        tpd_limit=body.tpd_limit,
    )


@router.get("/v1/keys", response_model=list[KeySummary])
async def list_api_keys(
    _: MasterKeyDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """List all API keys with aggregated usage stats."""
    keys_result = await db.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    keys = keys_result.scalars().all()

    summaries = []
    for key in keys:
        # Aggregate usage
        usage_result = await db.execute(
            select(
                func.count(UsageLog.id).label("total_requests"),
                func.coalesce(func.sum(UsageLog.cost_usd), 0.0).label("total_cost"),
            ).where(UsageLog.api_key_id == key.id)
        )
        row = usage_result.one()
        summaries.append(
            KeySummary(
                id=key.id,
                org_id=key.org_id,
                name=key.name,
                is_active=key.is_active,
                rpm_limit=key.rpm_limit,
                tpd_limit=key.tpd_limit,
                created_at=key.created_at.isoformat(),
                total_requests=row.total_requests or 0,
                total_cost_usd=round(row.total_cost or 0.0, 6),
            )
        )

    return summaries


@router.delete("/v1/keys/{key_id}", status_code=204)
async def deactivate_api_key(
    key_id: str,
    _: MasterKeyDep,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Deactivate (soft-delete) an API key."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(status_code=404, detail="API key not found.")

    api_key.is_active = False
    await db.flush()
