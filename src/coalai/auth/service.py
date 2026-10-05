"""
Authentication service.

Responsibility: verify an API key, return a verified TenantContext.

Flow:
  1. sha256(raw_key) → look up in Redis cache (fast path)
  2. On Redis miss or Redis unavailable → query PostgreSQL (fallback)
  3. On PostgreSQL also unavailable → raise COALAIError(SERVICE_UNAVAILABLE)
     The request is rejected with 503. We do NOT proceed without verified auth.
  4. Check key is not revoked and not expired.
  5. Return TenantContext populated from the database record.

Security invariants enforced here:
  - Raw key is NEVER logged or stored — only its sha256 hash.
  - TenantContext is ONLY populated from verified database records.
  - Callers MUST NOT pass untrusted request data into this module.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coalai.cache.redis_client import redis_get, redis_set
from coalai.config import settings
from coalai.db.models import ApiKey, Tenant
from coalai.models.contracts import TenantContext, TenantQuotas
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import get_logger

log = get_logger(__name__)

# Redis key namespace for API key → TenantContext cache.
_KEY_PREFIX = "coal:apikey:"


def _hash_key(raw_key: str) -> str:
    """sha256 of the raw API key. This is what is stored in the database."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _cache_key(key_hash: str) -> str:
    return f"{_KEY_PREFIX}{key_hash}"


def _tenant_context_to_json(ctx: TenantContext) -> str:
    return json.dumps(
        {
            "tenant_id": str(ctx.tenant_id),
            "api_key_id": str(ctx.api_key_id),
            "quotas": {
                "requests_per_minute": ctx.quotas.requests_per_minute,
                "requests_per_day": ctx.quotas.requests_per_day,
                "monthly_budget_usd": ctx.quotas.monthly_budget_usd,
            },
            "enabled_models": ctx.enabled_models,
        }
    )


def _tenant_context_from_json(data: str) -> TenantContext:
    d = json.loads(data)
    quotas_d = d["quotas"]
    return TenantContext(
        tenant_id=uuid.UUID(d["tenant_id"]),
        api_key_id=uuid.UUID(d["api_key_id"]),
        quotas=TenantQuotas(
            requests_per_minute=quotas_d["requests_per_minute"],
            requests_per_day=quotas_d["requests_per_day"],
            monthly_budget_usd=quotas_d.get("monthly_budget_usd"),
        ),
        enabled_models=d["enabled_models"],
    )


async def _load_from_db(
    key_hash: str,
    session: AsyncSession,
) -> TenantContext:
    """
    Load and verify an API key from PostgreSQL.
    Raises COALAIError on any failure condition.
    """
    stmt = (
        select(ApiKey, Tenant)
        .join(Tenant, ApiKey.tenant_id == Tenant.tenant_id)
        .where(ApiKey.key_hash == key_hash)
    )
    try:
        row: Any = (await session.execute(stmt)).one_or_none()
    except Exception as exc:
        log.error("auth_db_query_failed", error=str(exc))
        raise COALAIError(
            COALAIErrorType.SERVICE_UNAVAILABLE,
            "Authentication service unavailable",
        ) from exc

    if row is None:
        raise COALAIError(
            COALAIErrorType.AUTHENTICATION_FAILED,
            "Invalid API key",
        )

    api_key: ApiKey = row.ApiKey
    tenant: Tenant = row.Tenant

    # Check revocation
    if api_key.revoked:
        raise COALAIError(
            COALAIErrorType.AUTHENTICATION_FAILED,
            "API key has been revoked",
        )

    # Check expiry
    if api_key.expires_at is not None:
        now = datetime.now(tz=timezone.utc)
        expires = api_key.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if now > expires:
            raise COALAIError(
                COALAIErrorType.AUTHENTICATION_FAILED,
                "API key has expired",
            )

    # Check tenant status
    if tenant.status != "active":
        raise COALAIError(
            COALAIErrorType.AUTHORIZATION_FAILED,
            "Tenant account is not active",
        )

    return TenantContext(
        tenant_id=api_key.tenant_id,
        api_key_id=api_key.key_id,
        quotas=TenantQuotas(
            requests_per_minute=api_key.requests_per_minute,
            requests_per_day=api_key.requests_per_day,
        ),
        enabled_models=[],  # Phase 5+: load from routing_policies table
    )


async def verify_api_key(
    raw_key: str,
    session: AsyncSession,
) -> TenantContext:
    """
    Verify an API key and return the caller's TenantContext.

    This is the ONLY public entry point for authentication.
    All other auth-related code is private to this module.

    Raises COALAIError on any auth failure.
    Never logs the raw key. Logs only the key_hash prefix for traceability.
    """
    if not raw_key or not raw_key.startswith("coal_"):
        raise COALAIError(
            COALAIErrorType.AUTHENTICATION_FAILED,
            "Malformed API key",
        )

    key_hash = _hash_key(raw_key)
    log_key_prefix = key_hash[:8]  # Safe to log — not the raw key

    # 1. Try Redis cache (fast path)
    cached = await redis_get(_cache_key(key_hash), use_case="auth")
    if cached is not None:
        try:
            ctx = _tenant_context_from_json(cached)
            log.debug("auth_cache_hit", key_prefix=log_key_prefix)
            return ctx
        except Exception as exc:
            # Corrupted cache entry — fall through to DB
            log.warning("auth_cache_corrupted", key_prefix=log_key_prefix, error=str(exc))

    # 2. Query PostgreSQL (authoritative source)
    log.debug("auth_cache_miss_querying_db", key_prefix=log_key_prefix)
    ctx = await _load_from_db(key_hash, session)

    # 3. Write back to Redis cache (fire-and-forget — cache write failure is acceptable)
    await redis_set(
        _cache_key(key_hash),
        _tenant_context_to_json(ctx),
        ttl_seconds=settings.api_key_cache_ttl_seconds,
        use_case="auth",
    )

    log.info(
        "auth_verified",
        key_prefix=log_key_prefix,
        # tenant_id is safe to log here — it came from the verified DB record
        tenant_id=str(ctx.tenant_id),
    )
    return ctx


async def invalidate_key_cache(raw_key: str) -> None:
    """
    Remove a key from the Redis cache (e.g., on revocation).
    This ensures revoked keys are not served from cache.
    """
    key_hash = _hash_key(raw_key)
    from coalai.cache.redis_client import redis_delete

    await redis_delete(_cache_key(key_hash), use_case="auth")
