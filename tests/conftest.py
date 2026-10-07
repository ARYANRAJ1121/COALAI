"""
Shared pytest fixtures for all COALAI tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from coalai.models.contracts import Message, PipelineContext, TenantContext, TenantQuotas


@pytest.fixture
def tenant_id() -> uuid.UUID:
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def api_key_id() -> uuid.UUID:
    return uuid.UUID("87654321-4321-8765-4321-876543218765")


@pytest.fixture
def tenant_context(tenant_id: uuid.UUID, api_key_id: uuid.UUID) -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        quotas=TenantQuotas(requests_per_minute=60, requests_per_day=10000),
        enabled_models=[],
    )


@pytest.fixture
def pipeline_ctx(tenant_context: TenantContext) -> PipelineContext:
    return PipelineContext(
        request_id=uuid.uuid4(),
        trace_id="abc123",
        received_at=datetime.now(tz=timezone.utc),
        messages=[Message(role="user", content="Hello")],
        model_hint="llama3.2:3b",
        stream=False,
        tenant_context=tenant_context,
    )


@pytest.fixture
def mock_session() -> MagicMock:
    """A mock AsyncSession that accepts flush/commit/add."""
    session = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    return session
