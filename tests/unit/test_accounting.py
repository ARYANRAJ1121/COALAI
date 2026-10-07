"""
Unit tests for the durable accounting service (v0.3 Option A).

Tests:
  - write_confirmed: successful write → returns event_id
  - write_confirmed: DB failure → raises AccountingError
  - write_pending: successful write → returns event_id
  - write_pending: DB failure → raises AccountingError
  - confirm_pending: successful update → no exception
  - confirm_pending: DB failure → logs error but does NOT raise
    (PENDING row exists; reconciliation handles it)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from coalai.accounting.service import (
    AccountingError,
    confirm_pending,
    write_confirmed,
    write_pending,
)
from coalai.models.contracts import ExecutionResult, FinishReason
from coalai.models.errors import COALAIErrorType


def make_ctx_with_result(pipeline_ctx: MagicMock) -> MagicMock:
    """Build a PipelineContext mock with a valid result."""
    result = MagicMock(spec=ExecutionResult)
    result.model_used = "llama3.2:3b"
    result.provider_used = "ollama"
    result.input_tokens = 10
    result.output_tokens = 5
    result.finish_reason = FinishReason.STOP
    pipeline_ctx.result = result
    pipeline_ctx.estimated_cost_usd = 0.0
    pipeline_ctx.trace_id = "abc123"
    pipeline_ctx.exact_cache_hit = False
    return pipeline_ctx


# ── write_confirmed success ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_confirmed_success(pipeline_ctx: MagicMock, mock_session: MagicMock) -> None:
    """Successful accounting write → event_id returned, session.add called."""
    make_ctx_with_result(pipeline_ctx)

    event_id = await write_confirmed(pipeline_ctx, session=mock_session)

    assert isinstance(event_id, uuid.UUID)
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


# ── write_confirmed DB failure ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_confirmed_db_failure_raises(
    pipeline_ctx: MagicMock, mock_session: MagicMock
) -> None:
    """DB failure during CONFIRMED write → AccountingError raised (triggers 503)."""
    make_ctx_with_result(pipeline_ctx)
    mock_session.flush.side_effect = OperationalError("conn lost", {}, None)

    with pytest.raises(AccountingError):
        await write_confirmed(pipeline_ctx, session=mock_session)


# ── write_pending success ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_pending_success(pipeline_ctx: MagicMock, mock_session: MagicMock) -> None:
    """Successful PENDING write → event_id returned."""
    event_id = await write_pending(
        pipeline_ctx,
        estimated_input_tokens=20,
        session=mock_session,
    )
    assert isinstance(event_id, uuid.UUID)
    mock_session.add.assert_called_once()


# ── write_pending DB failure ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_pending_db_failure_raises(
    pipeline_ctx: MagicMock, mock_session: MagicMock
) -> None:
    """DB failure during PENDING write → AccountingError raised (triggers 503)."""
    mock_session.flush.side_effect = OperationalError("conn lost", {}, None)

    with pytest.raises(AccountingError):
        await write_pending(
            pipeline_ctx,
            estimated_input_tokens=20,
            session=mock_session,
        )


# ── confirm_pending success ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_pending_success(mock_session: MagicMock) -> None:
    """Successful confirm → no exception raised."""
    mock_session.execute = AsyncMock()
    event_id = uuid.uuid4()

    await confirm_pending(
        event_id,
        actual_input_tokens=10,
        actual_output_tokens=5,
        estimated_cost_usd=0.0,
        session=mock_session,
    )
    mock_session.execute.assert_called_once()


# ── confirm_pending failure does NOT raise ────────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_pending_failure_does_not_raise(mock_session: MagicMock) -> None:
    """
    DB failure during confirm → error is logged but NOT raised.

    The PENDING row already exists; reconciliation will promote it.
    The client has already received streaming chunks — we cannot return 503.
    """
    from sqlalchemy.exc import OperationalError

    mock_session.execute = AsyncMock(
        side_effect=OperationalError("conn lost", {}, None)
    )
    event_id = uuid.uuid4()

    # Must NOT raise
    await confirm_pending(
        event_id,
        actual_input_tokens=10,
        actual_output_tokens=5,
        estimated_cost_usd=0.0,
        session=mock_session,
    )
    # If we reach here, the test passes
