"""
Shared fixtures for the test suite.

Uses SQLite (aiosqlite) for database and a dict-based fake for Redis.
No real API keys are required.
"""
import asyncio
import hashlib
import os
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database import Base, get_db
from app.models.api_key import ApiKey
from app.config import get_settings

# Use in-memory SQLite for tests
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

settings = get_settings()


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine):
    SessionLocal = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session
        await session.rollback()


class FakeRedis:
    """In-memory Redis substitute for testing."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def ping(self):
        return True

    async def get(self, key: str):
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._store[key] = value

    async def delete(self, key: str):
        self._store.pop(key, None)

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

    async def incrby(self, key: str, amount: int) -> int:
        val = int(self._store.get(key, 0)) + amount
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, ttl: int):
        pass  # No-op in fake

    def pipeline(self):
        return FakePipeline(self)

    async def aclose(self):
        pass


class FakePipeline:
    """Minimal pipeline that runs commands sequentially."""

    def __init__(self, redis: FakeRedis):
        self._redis = redis
        self._commands: list = []

    def incr(self, key: str):
        self._commands.append(("incr", key))
        return self

    def incrby(self, key: str, amount: int):
        self._commands.append(("incrby", key, amount))
        return self

    def expire(self, key: str, ttl: int):
        self._commands.append(("expire", key, ttl))
        return self

    def get(self, key: str):
        self._commands.append(("get", key))
        return self

    async def execute(self):
        results = []
        for cmd in self._commands:
            op = cmd[0]
            if op == "incr":
                results.append(await self._redis.incr(cmd[1]))
            elif op == "incrby":
                results.append(await self._redis.incrby(cmd[1], cmd[2]))
            elif op == "expire":
                results.append(await self._redis.expire(cmd[1], cmd[2]))
            elif op == "get":
                results.append(await self._redis.get(cmd[1]))
        return results


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest_asyncio.fixture
async def test_api_key(db_session: AsyncSession) -> ApiKey:
    """Create a test API key in the DB."""
    raw_key = "gw-" + "a" * 64
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    api_key = ApiKey(
        id=str(uuid.uuid4()),
        key_hash=key_hash,
        org_id="test-org",
        name="Test Key",
        is_active=True,
        rpm_limit=60,
        tpd_limit=100_000,
    )
    db_session.add(api_key)
    await db_session.commit()
    return api_key


@pytest_asyncio.fixture
async def app_client(db_session: AsyncSession, fake_redis, test_api_key) -> AsyncGenerator:
    """FastAPI test client with overridden DB and Redis."""
    from app.main import create_app
    from app.services.cache import SemanticCache
    from app.services.rate_limiter import RateLimiter

    app = create_app()

    # Override database dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.state.redis = fake_redis
    app.state.rate_limiter = RateLimiter(fake_redis)
    app.state.cache = SemanticCache(fake_redis)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, test_api_key
