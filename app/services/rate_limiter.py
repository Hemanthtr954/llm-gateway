import logging
import time
from dataclasses import dataclass

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    allowed: bool
    remaining_rpm: int
    remaining_tpd: int
    reason: str = ""


class RateLimiter:
    """
    Redis-based sliding window rate limiter.
    Two limits:
      - RPM: requests per minute (rolling 60s window via INCR + EXPIRE)
      - TPD: tokens per day (rolling 24h window)

    Fail-open: if Redis is unreachable, allow the request.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    def _rpm_key(self, api_key_id: str) -> str:
        # 60-second bucket: key rotates every minute
        bucket = int(time.time()) // 60
        return f"rl:rpm:{api_key_id}:{bucket}"

    def _tpd_key(self, api_key_id: str) -> str:
        # Day bucket: key rotates every day (UTC)
        import datetime
        day = datetime.datetime.utcnow().strftime("%Y%m%d")
        return f"rl:tpd:{api_key_id}:{day}"

    async def check_and_increment(
        self,
        api_key_id: str,
        rpm_limit: int,
        tpd_limit: int,
        tokens_used: int = 0,
    ) -> RateLimitResult:
        """
        Check rate limits and increment counters atomically.
        tokens_used is for post-request TPD accounting (pass 0 for pre-request check).
        """
        try:
            rpm_key = self._rpm_key(api_key_id)
            tpd_key = self._tpd_key(api_key_id)

            pipe = self._redis.pipeline()
            pipe.incr(rpm_key)
            pipe.expire(rpm_key, 60)
            pipe.incrby(tpd_key, max(tokens_used, 1))
            pipe.expire(tpd_key, 86400)
            pipe.get(rpm_key)
            pipe.get(tpd_key)
            results = await pipe.execute()

            current_rpm = int(results[4] or 1)
            current_tpd = int(results[5] or 1)

            if current_rpm > rpm_limit:
                return RateLimitResult(
                    allowed=False,
                    remaining_rpm=0,
                    remaining_tpd=max(tpd_limit - current_tpd, 0),
                    reason=f"RPM limit exceeded ({current_rpm}/{rpm_limit})",
                )

            if current_tpd > tpd_limit:
                return RateLimitResult(
                    allowed=False,
                    remaining_rpm=max(rpm_limit - current_rpm, 0),
                    remaining_tpd=0,
                    reason=f"TPD limit exceeded ({current_tpd}/{tpd_limit})",
                )

            return RateLimitResult(
                allowed=True,
                remaining_rpm=max(rpm_limit - current_rpm, 0),
                remaining_tpd=max(tpd_limit - current_tpd, 0),
            )

        except Exception as exc:
            logger.error("Rate limiter Redis error (fail-open): %s", exc)
            # Fail open: allow the request if Redis is down
            return RateLimitResult(
                allowed=True,
                remaining_rpm=rpm_limit,
                remaining_tpd=tpd_limit,
                reason="rate_limiter_unavailable",
            )

    async def record_tokens(
        self,
        api_key_id: str,
        tokens: int,
    ) -> None:
        """Record token usage after a successful request."""
        if tokens <= 0:
            return
        try:
            tpd_key = self._tpd_key(api_key_id)
            pipe = self._redis.pipeline()
            pipe.incrby(tpd_key, tokens)
            pipe.expire(tpd_key, 86400)
            await pipe.execute()
        except Exception as exc:
            logger.error("Failed to record tokens in Redis: %s", exc)
