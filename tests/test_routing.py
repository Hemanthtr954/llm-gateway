"""
Tests for:
- Provider selection logic
- Fallback on provider failure
- Rate limit enforcement (mock Redis)
- Cache hit / miss
- Usage log written after successful request
"""
import hashlib
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.providers.base import ProviderResponse
from app.services.router import _pick_primary_provider, ProviderRouter
from app.services.rate_limiter import RateLimiter
from app.services.cache import SemanticCache
from app.services.usage_tracker import calculate_cost, track_usage
from app.models.usage_log import UsageLog


# ---------------------------------------------------------------------------
# Provider selection tests
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def test_claude_routes_to_anthropic(self):
        provider = _pick_primary_provider("claude-3-5-sonnet-20241022")
        assert provider.name == "anthropic"

    def test_claude_prefix_routes_to_anthropic(self):
        provider = _pick_primary_provider("claude-3-opus-20240229")
        assert provider.name == "anthropic"

    def test_llama_routes_to_groq(self):
        provider = _pick_primary_provider("llama-3.1-8b-instant")
        assert provider.name == "groq"

    def test_mixtral_routes_to_groq(self):
        provider = _pick_primary_provider("mixtral-8x7b-32768")
        assert provider.name == "groq"

    def test_gemma_routes_to_groq(self):
        provider = _pick_primary_provider("gemma2-9b-it")
        assert provider.name == "groq"

    def test_gpt4o_routes_to_openai(self):
        provider = _pick_primary_provider("gpt-4o")
        assert provider.name == "openai"

    def test_gpt35_turbo_routes_to_openai(self):
        provider = _pick_primary_provider("gpt-3.5-turbo")
        assert provider.name == "openai"

    def test_unknown_model_routes_to_openai(self):
        provider = _pick_primary_provider("some-future-model-xyz")
        assert provider.name == "openai"


# ---------------------------------------------------------------------------
# Fallback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestProviderFallback:
    async def test_falls_back_on_primary_failure(self):
        """When primary provider raises, next provider in chain is tried."""
        mock_response = ProviderResponse(
            content="Hello from fallback",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100,
            provider="openai",
        )

        router = ProviderRouter()
        messages = [{"role": "user", "content": "hi"}]

        # Patch the retry decorator to not retry, then test fallback
        with patch(
            "app.services.router._anthropic.chat_complete",
            new_callable=AsyncMock,
            side_effect=Exception("Anthropic is down"),
        ), patch(
            "app.services.router._openai.chat_complete",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result, provider_used = await router.route(
                messages, "claude-3-5-sonnet-20241022"
            )
            assert provider_used == "openai"
            assert result.content == "Hello from fallback"

    async def test_raises_when_all_providers_fail(self):
        """RuntimeError raised if all providers fail."""
        router = ProviderRouter()
        messages = [{"role": "user", "content": "hi"}]

        with patch(
            "app.services.router._openai.chat_complete",
            new_callable=AsyncMock,
            side_effect=Exception("OpenAI down"),
        ), patch(
            "app.services.router._anthropic.chat_complete",
            new_callable=AsyncMock,
            side_effect=Exception("Anthropic down"),
        ), patch(
            "app.services.router._groq.chat_complete",
            new_callable=AsyncMock,
            side_effect=Exception("Groq down"),
        ):
            with pytest.raises(RuntimeError, match="All providers exhausted"):
                await router.route(messages, "gpt-4o")


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRateLimiter:
    async def test_allows_request_within_limit(self, fake_redis):
        limiter = RateLimiter(fake_redis)
        result = await limiter.check_and_increment(
            api_key_id="key-1", rpm_limit=60, tpd_limit=100_000
        )
        assert result.allowed is True
        assert result.remaining_rpm == 59

    async def test_blocks_request_over_rpm(self, fake_redis):
        limiter = RateLimiter(fake_redis)
        # Pre-fill RPM counter to limit
        for _ in range(60):
            await limiter.check_and_increment(
                api_key_id="key-2", rpm_limit=60, tpd_limit=100_000
            )

        # 61st request should be blocked
        result = await limiter.check_and_increment(
            api_key_id="key-2", rpm_limit=60, tpd_limit=100_000
        )
        assert result.allowed is False
        assert result.remaining_rpm == 0
        assert "RPM" in result.reason

    async def test_fails_open_when_redis_down(self):
        """When Redis is unreachable, requests are allowed (fail-open)."""
        from tests.conftest import FakeRedis

        class BrokenRedis(FakeRedis):
            def pipeline(self):
                raise ConnectionError("Redis is gone")

        limiter = RateLimiter(BrokenRedis())
        result = await limiter.check_and_increment(
            api_key_id="key-3", rpm_limit=60, tpd_limit=100_000
        )
        assert result.allowed is True
        assert "unavailable" in result.reason

    async def test_tpd_limit_enforced(self, fake_redis):
        limiter = RateLimiter(fake_redis)
        # Simulate large token usage
        import datetime
        day = datetime.datetime.utcnow().strftime("%Y%m%d")
        tpd_key = f"rl:tpd:key-4:{day}"
        await fake_redis.incrby(tpd_key, 99_999)

        result = await limiter.check_and_increment(
            api_key_id="key-4", rpm_limit=60, tpd_limit=100_000, tokens_used=10
        )
        assert result.allowed is False
        assert result.remaining_tpd == 0


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSemanticCache:
    async def test_cache_miss_returns_none(self, fake_redis):
        cache = SemanticCache(fake_redis)
        messages = [{"role": "user", "content": "hello"}]
        result = await cache.get(messages, "gpt-4o")
        assert result is None

    async def test_cache_hit_returns_response(self, fake_redis):
        cache = SemanticCache(fake_redis)
        messages = [{"role": "user", "content": "hello"}]
        model = "gpt-4o"
        response = ProviderResponse(
            content="Hi there!",
            model=model,
            prompt_tokens=5,
            completion_tokens=3,
            latency_ms=200,
            provider="openai",
        )

        await cache.store(messages, model, response)
        hit = await cache.get(messages, model)

        assert hit is not None
        assert hit.content == "Hi there!"
        assert hit.prompt_tokens == 5

    async def test_different_messages_different_keys(self, fake_redis):
        cache = SemanticCache(fake_redis)
        msg1 = [{"role": "user", "content": "hello"}]
        msg2 = [{"role": "user", "content": "goodbye"}]
        model = "gpt-4o"

        response = ProviderResponse(
            content="Hi!", model=model, prompt_tokens=5,
            completion_tokens=3, latency_ms=100, provider="openai"
        )
        await cache.store(msg1, model, response)

        result = await cache.get(msg2, model)
        assert result is None

    async def test_different_models_different_keys(self, fake_redis):
        cache = SemanticCache(fake_redis)
        messages = [{"role": "user", "content": "hello"}]

        response = ProviderResponse(
            content="Hi!", model="gpt-4o", prompt_tokens=5,
            completion_tokens=3, latency_ms=100, provider="openai"
        )
        await cache.store(messages, "gpt-4o", response)

        result = await cache.get(messages, "gpt-4o-mini")
        assert result is None


# ---------------------------------------------------------------------------
# Cost calculation tests
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_gpt4o_cost(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
        # 1000/1000 * 0.0025 + 1000/1000 * 0.01 = 0.0125
        assert abs(cost - 0.0125) < 1e-6

    def test_gpt4o_mini_cost(self):
        cost = calculate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
        # 0.00015 + 0.0006 = 0.00075
        assert abs(cost - 0.00075) < 1e-6

    def test_claude_sonnet_cost(self):
        cost = calculate_cost(
            "claude-3-5-sonnet-20241022", prompt_tokens=1000, completion_tokens=1000
        )
        # 0.003 + 0.015 = 0.018
        assert abs(cost - 0.018) < 1e-6

    def test_groq_llama_cost(self):
        cost = calculate_cost("llama-3.1-8b-instant", prompt_tokens=1000, completion_tokens=1000)
        # 0.00005 + 0.0001 = 0.00015
        assert abs(cost - 0.00015) < 1e-6

    def test_unknown_model_uses_default(self):
        cost = calculate_cost("some-unknown-model-xyz", prompt_tokens=1000, completion_tokens=1000)
        assert cost > 0  # Falls back to default pricing


# ---------------------------------------------------------------------------
# Usage logging tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestUsageLogging:
    async def test_usage_log_written(self, db_session, test_api_key):
        log = await track_usage(
            db=db_session,
            api_key_id=test_api_key.id,
            org_id=test_api_key.org_id,
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=300,
            status="success",
        )
        await db_session.commit()

        result = await db_session.execute(
            select(UsageLog).where(UsageLog.id == log.id)
        )
        fetched = result.scalar_one()

        assert fetched.provider == "openai"
        assert fetched.model == "gpt-4o"
        assert fetched.prompt_tokens == 100
        assert fetched.completion_tokens == 50
        assert fetched.total_tokens == 150
        assert fetched.cost_usd > 0
        assert fetched.status == "success"

    async def test_error_status_logged(self, db_session, test_api_key):
        log = await track_usage(
            db=db_session,
            api_key_id=test_api_key.id,
            org_id=test_api_key.org_id,
            provider="unknown",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            status="error",
        )
        await db_session.commit()

        result = await db_session.execute(
            select(UsageLog).where(UsageLog.id == log.id)
        )
        fetched = result.scalar_one()
        assert fetched.status == "error"
        assert fetched.cost_usd == 0.0
