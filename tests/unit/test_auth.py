"""
Unit tests for the Auth module.

Tests:
  - Valid key → TenantContext returned
  - Invalid key → AUTHENTICATION_FAILED
  - Revoked key → AUTHENTICATION_FAILED
  - Malformed key (wrong prefix) → AUTHENTICATION_FAILED
  - Redis cache hit → returns without hitting DB
  - Redis miss → falls back to Postgres
  - Both Redis and Postgres unavailable → SERVICE_UNAVAILABLE (503)

Provider mocking strategy:
  - Redis: fakeredis async client
  - Postgres: MagicMock AsyncSession with controlled return values
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coalai.auth.service import _hash_key, _tenant_context_to_json, verify_api_key
from coalai.models.contracts import TenantContext, TenantQuotas
from coalai.models.errors import COALAIError, COALAIErrorType


def make_api_key_row(
    tenant_id: uuid.UUID,
    key_id: uuid.UUID,
    key_hash: str,
    revoked: bool = False,
    expires_at: datetime | None = None,
) -> MagicMock:
    """Build a mock SQLAlchemy row matching (ApiKey, Tenant)."""
    api_key = MagicMock()
    api_key.key_id = key_id
    api_key.tenant_id = tenant_id
    api_key.key_hash = key_hash
    api_key.revoked = revoked
    api_key.expires_at = expires_at
    api_key.requests_per_minute = 60
    api_key.requests_per_day = 10000

    tenant = MagicMock()
    tenant.tenant_id = tenant_id
    tenant.status = "active"

    row = MagicMock()
    row.ApiKey = api_key
    row.Tenant = tenant
    return row


RAW_KEY = "coal_sk_testkey1234567890abcdefghijklmnopqrstuvwx"
KEY_HASH = _hash_key(RAW_KEY)
TENANT_ID = uuid.uuid4()
KEY_ID = uuid.uuid4()


# ── Valid key, DB lookup ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_valid_key_db_lookup(mock_session: MagicMock) -> None:
    """Valid key with Redis miss → fetches from Postgres → returns TenantContext."""
    row = make_api_key_row(TENANT_ID, KEY_ID, KEY_HASH)
    mock_session.execute.return_value.one_or_none = MagicMock(return_value=row)

    with patch("coalai.auth.service.redis_get", new=AsyncMock(return_value=None)), \
         patch("coalai.auth.service.redis_set", new=AsyncMock(return_value=True)):
        ctx = await verify_api_key(RAW_KEY, mock_session)

    assert isinstance(ctx, TenantContext)
    assert ctx.tenant_id == TENANT_ID
    assert ctx.api_key_id == KEY_ID
    assert ctx.quotas.requests_per_minute == 60


# ── Valid key, Redis cache hit ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_valid_key_cache_hit(mock_session: MagicMock) -> None:
    """Valid key with Redis hit → returns without touching Postgres."""
    expected_ctx = TenantContext(
        tenant_id=TENANT_ID,
        api_key_id=KEY_ID,
        quotas=TenantQuotas(requests_per_minute=60, requests_per_day=10000),
        enabled_models=[],
    )
    cached_json = _tenant_context_to_json(expected_ctx)

    with patch("coalai.auth.service.redis_get", new=AsyncMock(return_value=cached_json)):
        ctx = await verify_api_key(RAW_KEY, mock_session)

    assert ctx.tenant_id == TENANT_ID
    # Postgres was NOT called
    mock_session.execute.assert_not_called()


# ── Invalid key ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_invalid_key_not_in_db(mock_session: MagicMock) -> None:
    """Key hash not found in DB → AUTHENTICATION_FAILED."""
    mock_session.execute.return_value.one_or_none = MagicMock(return_value=None)

    with patch("coalai.auth.service.redis_get", new=AsyncMock(return_value=None)), \
         patch("coalai.auth.service.redis_set", new=AsyncMock(return_value=False)):
        with pytest.raises(COALAIError) as exc_info:
            await verify_api_key(RAW_KEY, mock_session)

    assert exc_info.value.error_type == COALAIErrorType.AUTHENTICATION_FAILED
    assert exc_info.value.http_status == 401


# ── Malformed key (wrong prefix) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_malformed_key(mock_session: MagicMock) -> None:
    """Key without coal_ prefix → AUTHENTICATION_FAILED before any DB call."""
    with pytest.raises(COALAIError) as exc_info:
        await verify_api_key("sk-openai-abc123", mock_session)

    assert exc_info.value.error_type == COALAIErrorType.AUTHENTICATION_FAILED
    mock_session.execute.assert_not_called()


# ── Revoked key ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_revoked_key(mock_session: MagicMock) -> None:
    """Revoked key → AUTHENTICATION_FAILED."""
    row = make_api_key_row(TENANT_ID, KEY_ID, KEY_HASH, revoked=True)
    mock_session.execute.return_value.one_or_none = MagicMock(return_value=row)

    with patch("coalai.auth.service.redis_get", new=AsyncMock(return_value=None)), \
         patch("coalai.auth.service.redis_set", new=AsyncMock(return_value=True)):
        with pytest.raises(COALAIError) as exc_info:
            await verify_api_key(RAW_KEY, mock_session)

    assert exc_info.value.error_type == COALAIErrorType.AUTHENTICATION_FAILED


# ── Expired key ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_expired_key(mock_session: MagicMock) -> None:
    """Expired key → AUTHENTICATION_FAILED."""
    past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    row = make_api_key_row(TENANT_ID, KEY_ID, KEY_HASH, expires_at=past)
    mock_session.execute.return_value.one_or_none = MagicMock(return_value=row)

    with patch("coalai.auth.service.redis_get", new=AsyncMock(return_value=None)), \
         patch("coalai.auth.service.redis_set", new=AsyncMock(return_value=True)):
        with pytest.raises(COALAIError) as exc_info:
            await verify_api_key(RAW_KEY, mock_session)

    assert exc_info.value.error_type == COALAIErrorType.AUTHENTICATION_FAILED


# ── Postgres unavailable ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_postgres_unavailable(mock_session: MagicMock) -> None:
    """Redis miss + Postgres error → SERVICE_UNAVAILABLE (503)."""
    from sqlalchemy.exc import OperationalError

    mock_session.execute.side_effect = OperationalError("conn refused", {}, None)

    with patch("coalai.auth.service.redis_get", new=AsyncMock(return_value=None)):
        with pytest.raises(COALAIError) as exc_info:
            await verify_api_key(RAW_KEY, mock_session)

    assert exc_info.value.error_type == COALAIErrorType.SERVICE_UNAVAILABLE
    assert exc_info.value.http_status == 503
