"""
Redis client wrapper.

All Redis access goes through this module. No other module imports redis directly.

Failure policy per use case (v0.3 §11):
  - Rate limiting:    fail open (allow request, emit metric)
  - Exact cache:      fail open (treat as miss)
  - Circuit breaker:  fail open (assume CLOSED)
  - Auth key cache:   fail open (fall back to Postgres)
  - Budget tracking:  degraded mode (not enforced in Milestone 1)

Every method catches Redis errors and applies the appropriate policy.
Callers receive a typed result or a sentinel indicating unavailability —
they never see a Redis exception.
"""

from __future__ import annotations

import asyncio
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from coalai.config import settings
from coalai.observability.logging import get_logger

log = get_logger(__name__)

_client: Redis | None = None


def init_redis() -> None:
    """Create the Redis client. Called once at application startup."""
    global _client
    _client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=settings.redis_timeout_seconds,
        socket_timeout=settings.redis_timeout_seconds,
    )
    log.info("redis_client_initialised", url=settings.redis_url)


async def close_redis() -> None:
    """Close the Redis connection. Called at application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        log.info("redis_client_closed")
        _client = None


def _get_client() -> Redis:
    if _client is None:
        raise RuntimeError("Redis not initialised. Call init_redis() first.")
    return _client


async def redis_get(key: str, *, use_case: str) -> str | None:
    """
    GET a key. Returns None on cache miss OR on Redis failure (fail open).
    use_case is included in the failure log for observability.
    """
    try:
        return await asyncio.wait_for(
            _get_client().get(key),
            timeout=settings.redis_timeout_seconds,
        )
    except (RedisError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
        log.warning(
            "redis_get_failed",
            use_case=use_case,
            key=key,
            error=str(exc),
        )
        return None


async def redis_set(
    key: str,
    value: str,
    *,
    ttl_seconds: int,
    use_case: str,
) -> bool:
    """
    SET a key with TTL. Returns True on success, False on failure (fail open).
    """
    try:
        await asyncio.wait_for(
            _get_client().set(key, value, ex=ttl_seconds),
            timeout=settings.redis_timeout_seconds,
        )
        return True
    except (RedisError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
        log.warning(
            "redis_set_failed",
            use_case=use_case,
            key=key,
            error=str(exc),
        )
        return False


async def redis_delete(key: str, *, use_case: str) -> None:
    """DELETE a key. Failure is logged and swallowed (fail open)."""
    try:
        await asyncio.wait_for(
            _get_client().delete(key),
            timeout=settings.redis_timeout_seconds,
        )
    except (RedisError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
        log.warning("redis_delete_failed", use_case=use_case, key=key, error=str(exc))


async def redis_incr_with_expire(
    key: str,
    *,
    ttl_seconds: int,
    use_case: str,
) -> int | None:
    """
    Atomically increment a counter and set its TTL if it is new.
    Returns the new counter value, or None on Redis failure.
    Used for rate limiting.
    """
    try:
        pipe = _get_client().pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_seconds)
        results: list[Any] = await asyncio.wait_for(
            pipe.execute(),
            timeout=settings.redis_timeout_seconds,
        )
        return int(results[0])
    except (RedisError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
        log.warning(
            "redis_incr_failed",
            use_case=use_case,
            key=key,
            error=str(exc),
        )
        return None


async def redis_ping() -> bool:
    """Health check. Returns True if Redis is reachable."""
    try:
        result = await asyncio.wait_for(
            _get_client().ping(),
            timeout=settings.redis_timeout_seconds,
        )
        return bool(result)
    except Exception:
        return False
