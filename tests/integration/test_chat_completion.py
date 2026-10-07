"""
Integration test — full Iron Path.

Tests the complete request flow:
  client → COALAI gateway → Ollama (mocked) → COALAI → accounting → client

What this test proves:
  1. A valid API key authenticates correctly
  2. A well-formed request is processed end-to-end
  3. The response is OpenAI-compatible with COALAI extensions
  4. The accounting_ledger has exactly one row with correct token counts
  5. An invalid API key returns 401
  6. With Ollama "down" (connection refused), a structured error is returned

Setup:
  - Requires a running PostgreSQL instance (via COALAI_TEST_DB_URL env var
    or Docker Compose). Skipped if unavailable.
  - Ollama is always mocked — no real LLM calls in CI.

To run locally with Docker Compose test stack:
    docker compose -f docker-compose.test.yml up -d
    pytest tests/integration/ -v
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
import respx
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from coalai.db.models import AccountingLedger, ApiKey, Tenant, Base
from coalai.main import create_app

# ── Skip marker ───────────────────────────────────────────────────────────────

TEST_DB_URL = os.getenv(
    "COALAI_TEST_DB_URL",
    "postgresql+asyncpg://coalai:coalai@localhost:5432/coalai_test",
)

pytestmark = pytest.mark.integration


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="module")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Test database not available — skipping integration tests")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def provisioned_key(db_session: AsyncSession):
    """Insert a test tenant + API key into the test database."""
    raw_key = "coal_sk_integrationtestkey12345678901234567890"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    tenant_id = uuid.uuid4()
    key_id = uuid.uuid4()

    tenant = Tenant(tenant_id=tenant_id, name="integration-test-tenant", status="active")
    api_key = ApiKey(
        key_id=key_id,
        tenant_id=tenant_id,
        key_hash=key_hash,
        label="integration",
        requests_per_minute=1000,
        requests_per_day=100000,
    )
    db_session.add(tenant)
    db_session.add(api_key)
    await db_session.commit()

    return {"raw_key": raw_key, "tenant_id": tenant_id, "key_id": key_id}


# ── Tests ──────────────────────────────────────────────────────────────────────


MOCK_OLLAMA_RESPONSE = {
    "id": "chatcmpl-inttest-001",
    "object": "chat.completion",
    "model": "llama3.2:3b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 7, "total_tokens": 15},
}


@pytest.mark.asyncio
@respx.mock
async def test_full_iron_path(provisioned_key: dict, db_session: AsyncSession) -> None:
    """
    Full iron path: auth → validate → execute (mocked Ollama) → account → respond.

    Assertions:
      - HTTP 200 returned
      - Response is OpenAI-compatible
      - coalai extension fields present
      - accounting_ledger has exactly one CONFIRMED row
      - Token counts in ledger match provider response
    """
    respx.post("http://ollama:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=MOCK_OLLAMA_RESPONSE)
    )

    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "model": "llama3.2:3b"},
            headers={"Authorization": f"Bearer {provisioned_key['raw_key']}"},
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # OpenAI-compatible fields
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
    assert data["usage"]["total_tokens"] == 15

    # COALAI extension
    assert "coalai" in data
    assert data["coalai"]["provider_used"] == "ollama"
    assert data["coalai"]["cache_hit"] is False
    assert "request_id" in data["coalai"]

    request_id = uuid.UUID(data["coalai"]["request_id"])

    # Accounting ledger has exactly one CONFIRMED row
    result = await db_session.execute(
        select(AccountingLedger).where(AccountingLedger.request_id == request_id)
    )
    ledger_row = result.scalar_one_or_none()

    assert ledger_row is not None, "Accounting ledger row missing — invariant violated"
    assert ledger_row.status == "CONFIRMED"
    assert ledger_row.input_tokens == 8
    assert ledger_row.output_tokens == 7


@pytest.mark.asyncio
async def test_invalid_api_key_returns_401() -> None:
    """Request with an invalid API key returns 401."""
    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer coal_sk_thiskeyisnotvalid"},
        )

    assert response.status_code == 401
    data = response.json()
    assert data["error"]["error_type"] == "authentication_failed"


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401() -> None:
    """Request with no Authorization header returns 401."""
    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_provider_timeout_returns_structured_error(provisioned_key: dict) -> None:
    """When Ollama times out, COALAI returns a structured 504 error — not a crash."""
    import asyncio

    async def slow(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(60)
        return httpx.Response(200, json=MOCK_OLLAMA_RESPONSE)

    respx.post("http://ollama:11434/v1/chat/completions").mock(side_effect=slow)

    # Use a very short timeout for this test
    import coalai.config as cfg
    original_timeout = cfg.settings.provider_timeout_seconds
    cfg.settings.provider_timeout_seconds = 0.05

    app = create_app()
    try:
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {provisioned_key['raw_key']}"},
                timeout=30.0,
            )
    finally:
        cfg.settings.provider_timeout_seconds = original_timeout

    # Must be a structured error, not a crash
    assert response.status_code in (504, 503)
    data = response.json()
    assert "error" in data
    assert data["error"]["error_type"] in ("provider_timeout", "all_providers_failed")
