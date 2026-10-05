"""
Durable usage accounting service.

Implements v0.3 Option A: strong consistency accounting.

Invariant enforced here:
  "If COALAI returns a successful HTTP response, a durable accounting
   record for that request EXISTS in PostgreSQL."

This is implemented as a two-phase write for streaming:

  Non-streaming:
    1. Provider responds with complete result
    2. INSERT accounting_ledger (status=CONFIRMED) ← awaited
    3. If INSERT fails → raise AccountingError → gateway returns 503
    4. Return HTTP 200 to client

  Streaming:
    Phase 1 (before first chunk):
      1. INSERT accounting_ledger (status=PENDING, output_tokens=0)
      2. If INSERT fails → raise AccountingError → gateway returns 503 (no chunk sent yet)
      3. Begin streaming to client
    Phase 2 (after stream completes/cancels):
      1. UPDATE accounting_ledger SET status=CONFIRMED, actual token counts
      2. If UPDATE fails → log error + emit metric (row EXISTS; reconciliation handles it)

The reconciliation job promotes PENDING → CONFIRMED rows. It NEVER creates
new rows — a missing row means the request returned 503, not a data loss.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from coalai.db.models import AccountingLedger
from coalai.models.contracts import PipelineContext, utcnow
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import get_logger

log = get_logger(__name__)


class AccountingError(Exception):
    """Raised when a critical accounting write fails. Triggers a 503 response."""


async def write_confirmed(
    ctx: PipelineContext,
    *,
    session: AsyncSession,
) -> uuid.UUID:
    """
    Write a CONFIRMED accounting record for a completed non-streaming request.

    Called AFTER the provider has responded but BEFORE the HTTP response
    is sent to the client. If this raises, the gateway MUST return 503.

    Returns the new event_id.
    Raises AccountingError on any database failure.
    """
    if ctx.result is None:
        raise ValueError("Cannot write accounting: PipelineContext has no result")

    event_id = uuid.uuid4()
    now = utcnow()

    record = AccountingLedger(
        event_id=event_id,
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_context.tenant_id,  # type: ignore[union-attr]
        model_id=ctx.result.model_used,
        provider=ctx.result.provider_used,
        input_tokens=ctx.result.input_tokens,
        output_tokens=ctx.result.output_tokens,
        estimated_cost_usd=ctx.estimated_cost_usd,
        cache_hit=ctx.exact_cache_hit if hasattr(ctx, "exact_cache_hit") else False,
        trace_id=ctx.trace_id,
        status="CONFIRMED",
        created_at=now,
        completed_at=now,
    )

    try:
        session.add(record)
        await session.flush()  # Write to DB within the current transaction
        log.info(
            "accounting_confirmed_written",
            request_id=str(ctx.request_id),
            event_id=str(event_id),
            input_tokens=ctx.result.input_tokens,
            output_tokens=ctx.result.output_tokens,
            tenant_id=str(ctx.tenant_context.tenant_id),  # type: ignore[union-attr]
        )
        return event_id
    except SQLAlchemyError as exc:
        log.error(
            "accounting_write_failed",
            request_id=str(ctx.request_id),
            error=str(exc),
            tenant_id=str(ctx.tenant_context.tenant_id) if ctx.tenant_context else None,  # type: ignore[union-attr]
        )
        raise AccountingError(f"Accounting write failed: {exc}") from exc


async def write_pending(
    ctx: PipelineContext,
    *,
    estimated_input_tokens: int,
    session: AsyncSession,
) -> uuid.UUID:
    """
    Write a PENDING accounting record before streaming begins.

    Called BEFORE the first SSE chunk is sent to the client.
    If this raises, the gateway MUST return 503 (no chunk has been sent).

    Returns the new event_id (used by confirm_pending to update the record).
    Raises AccountingError on any database failure.
    """
    event_id = uuid.uuid4()
    now = utcnow()

    record = AccountingLedger(
        event_id=event_id,
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_context.tenant_id,  # type: ignore[union-attr]
        model_id=ctx.model_hint or "unknown",
        provider="ollama",  # Phase 5: use route.provider
        input_tokens=estimated_input_tokens,
        output_tokens=0,  # Not yet known; updated in Phase 2
        estimated_cost_usd=0.0,  # Updated in Phase 2
        cache_hit=False,
        trace_id=ctx.trace_id,
        status="PENDING",
        created_at=now,
        completed_at=None,
    )

    try:
        session.add(record)
        await session.flush()
        log.info(
            "accounting_pending_written",
            request_id=str(ctx.request_id),
            event_id=str(event_id),
            estimated_input_tokens=estimated_input_tokens,
            tenant_id=str(ctx.tenant_context.tenant_id),  # type: ignore[union-attr]
        )
        return event_id
    except SQLAlchemyError as exc:
        log.error(
            "accounting_pending_write_failed",
            request_id=str(ctx.request_id),
            error=str(exc),
        )
        raise AccountingError(f"Accounting pending write failed: {exc}") from exc


async def confirm_pending(
    event_id: uuid.UUID,
    *,
    actual_input_tokens: int,
    actual_output_tokens: int,
    estimated_cost_usd: float = 0.0,
    session: AsyncSession,
    request_id: str = "",
) -> None:
    """
    Promote a PENDING accounting record to CONFIRMED with actual token counts.

    Called AFTER a stream completes. Failure here is NOT fatal — the PENDING
    row already exists. Log the error and emit a metric; the reconciliation
    job will pick it up.
    """
    try:
        stmt = (
            update(AccountingLedger)
            .where(AccountingLedger.event_id == event_id)
            .values(
                status="CONFIRMED",
                output_tokens=actual_output_tokens,
                input_tokens=actual_input_tokens,
                estimated_cost_usd=estimated_cost_usd,
                completed_at=utcnow(),
            )
        )
        await session.execute(stmt)
        log.info(
            "accounting_pending_confirmed",
            event_id=str(event_id),
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
        )
    except SQLAlchemyError as exc:
        # The PENDING row exists. Reconciliation will promote it.
        # This is NOT a 503 — the client has already received the stream.
        log.error(
            "accounting_confirm_failed",
            event_id=str(event_id),
            request_id=request_id,
            error=str(exc),
        )
        # TODO Phase 4: metrics.increment("coalai_accounting_confirm_failures_total")
