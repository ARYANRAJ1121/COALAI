"""
Request execution pipeline.

This module orchestrates the Milestone 1 "Iron Path":
  Receive → Build context → Auth → Rate-limit → Validate →
  Execute → Account → Respond

Each stage is a discrete function that reads from / writes to the
PipelineContext. No stage contains HTTP concerns (that lives in router.py).
No stage contains provider-specific logic (that lives in providers/).

Pipeline design principles:
  - Each stage produces a clearly defined side effect on PipelineContext.
  - Failure in any critical stage raises COALAIError (never raw exceptions).
  - Auth must complete before any tenant_id appears in logs.
  - Accounting write (Option A) happens before the response is sent.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from coalai.accounting.service import (
    AccountingError,
    confirm_pending,
    write_confirmed,
    write_pending,
)
from coalai.auth.service import verify_api_key
from coalai.cache.redis_client import redis_incr_with_expire
from coalai.config import settings
from coalai.gateway.schemas import ChatCompletionRequest
from coalai.models.contracts import (
    Message,
    NormalizedChunk,
    NormalizedRequest,
    PipelineContext,
    TenantContext,
    utcnow,
)
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import bind_request_context, get_logger
from coalai.providers.ollama import OllamaProvider
from coalai.reliability.retry import RetryPolicy, retry_with_backoff

log = get_logger(__name__)

_RATE_LIMIT_WINDOW_SECONDS = 60  # 1-minute sliding window


# ── Stage 1: Build pipeline context from HTTP request ─────────────────────────


def build_context(req: ChatCompletionRequest) -> PipelineContext:
    """
    Create a new PipelineContext from the parsed HTTP request body.
    At this point, tenant identity is not yet known.
    """
    request_id = uuid.uuid4()
    trace_id = str(uuid.uuid4()).replace("-", "")  # Phase 4: use OTEL trace ID

    # Log request received with null tenant_id (pre-auth — v0.3 FIX 3)
    bind_request_context(
        request_id=str(request_id),
        trace_id=trace_id,
        tenant_id=None,  # MUST be None until auth completes
    )
    log.info("request_received", path="/v1/chat/completions", stream=req.stream)

    messages = [Message(role=m.role, content=m.content) for m in req.messages]

    return PipelineContext(
        request_id=request_id,
        trace_id=trace_id,
        received_at=utcnow(),
        messages=messages,
        model_hint=req.model,
        stream=req.stream,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=req.stop,
        seed=req.seed,
        idempotency_key=req.idempotency_key,
    )


# ── Stage 2: Authentication ────────────────────────────────────────────────────


async def run_auth(
    ctx: PipelineContext,
    raw_key: str,
    session: AsyncSession,
) -> None:
    """
    Verify the API key and populate ctx.tenant_context.
    After this stage, bind tenant_id to the log context.

    Raises COALAIError on auth failure.
    """
    tenant_ctx: TenantContext = await verify_api_key(raw_key, session)
    ctx.tenant_context = tenant_ctx

    # Now that tenant identity is verified, bind tenant_id to log context
    bind_request_context(
        request_id=str(ctx.request_id),
        trace_id=ctx.trace_id,
        tenant_id=str(tenant_ctx.tenant_id),
    )
    log.debug("auth_completed", api_key_id=str(tenant_ctx.api_key_id))


# ── Stage 3: Rate limiting ─────────────────────────────────────────────────────


async def run_rate_limit(ctx: PipelineContext) -> None:
    """
    Enforce per-tenant request rate limits using Redis atomic counters.

    Fail-open policy (v0.3 §11): if Redis is unavailable, allow the request
    and emit a warning. Never reject a request because Redis is down.

    Raises COALAIError(RATE_LIMITED) if the tenant's RPM quota is exceeded.
    """
    assert ctx.tenant_context is not None
    tenant_id = str(ctx.tenant_context.tenant_id)
    rpm_limit = ctx.tenant_context.quotas.requests_per_minute

    # Compute a 1-minute window key
    window = int(utcnow().timestamp()) // _RATE_LIMIT_WINDOW_SECONDS
    key = f"coal:ratelimit:{tenant_id}:{window}"

    count = await redis_incr_with_expire(
        key,
        ttl_seconds=_RATE_LIMIT_WINDOW_SECONDS * 2,  # 2x window to handle boundary
        use_case="rate_limit",
    )

    if count is None:
        # Redis unavailable → fail open
        log.warning("rate_limit_redis_unavailable_fail_open", tenant_id=tenant_id)
        return

    if count > rpm_limit:
        log.warning(
            "rate_limit_exceeded",
            tenant_id=tenant_id,
            count=count,
            limit=rpm_limit,
        )
        raise COALAIError(
            COALAIErrorType.RATE_LIMITED,
            f"Rate limit exceeded: {count}/{rpm_limit} requests per minute",
            http_status=429,
        )


# ── Stage 4: Resolve model ────────────────────────────────────────────────────


def resolve_model(ctx: PipelineContext) -> str:
    """
    Determine the actual model to use for this request.

    Milestone 1: no routing logic. Use the model_hint from the request,
    or fall back to the configured default. Phase 5 adds the full Router.
    """
    model = ctx.model_hint or settings.ollama_default_model
    log.debug("model_resolved", model=model, hint=ctx.model_hint)
    return model


# ── Stage 5: Execute (non-streaming) ──────────────────────────────────────────


async def execute_completion(
    ctx: PipelineContext,
    model: str,
    provider: OllamaProvider,
    policy: RetryPolicy,
) -> None:
    """
    Call the provider with retry. Populates ctx.result and ctx.all_attempts.
    Raises COALAIError if all attempts are exhausted.
    """
    normalized_req = NormalizedRequest(
        messages=ctx.messages,
        model=model,
        stream=False,
        request_id=ctx.request_id,
        max_tokens=ctx.max_tokens,
        temperature=ctx.temperature,
        stop=ctx.stop,
        seed=ctx.seed,
    )

    attempt_number = 0

    async def _call():
        nonlocal attempt_number
        attempt_number += 1
        result = await provider.complete(
            normalized_req,
            timeout_seconds=settings.provider_timeout_seconds,
        )
        return result

    result = await retry_with_backoff(
        _call,
        policy=policy,
        request_id=str(ctx.request_id),
    )

    ctx.result = result
    ctx.all_attempts.append(result)

    log.info(
        "execution_completed",
        model=model,
        provider=provider.provider_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=round(result.provider_latency_ms, 1),
    )


# ── Stage 5 (streaming variant) ───────────────────────────────────────────────


async def execute_stream(
    ctx: PipelineContext,
    model: str,
    provider: OllamaProvider,
) -> AsyncIterator[NormalizedChunk]:
    """
    Execute a streaming provider call.
    Returns an async iterator of NormalizedChunk objects.
    The caller (router.py) handles accounting around this iterator.
    """
    normalized_req = NormalizedRequest(
        messages=ctx.messages,
        model=model,
        stream=True,
        request_id=ctx.request_id,
        max_tokens=ctx.max_tokens,
        temperature=ctx.temperature,
        stop=ctx.stop,
        seed=ctx.seed,
    )

    # Streaming retries are not applied at the chunk level in Milestone 1.
    # If streaming fails before the first chunk, the accounting PENDING write
    # has not been committed yet — the caller should not retry from mid-stream.
    return provider.stream(
        normalized_req,
        timeout_seconds=settings.provider_streaming_timeout_seconds,
    )


# ── Stage 6: Account (non-streaming) ──────────────────────────────────────────


async def run_accounting_confirmed(
    ctx: PipelineContext,
    session: AsyncSession,
) -> None:
    """
    Write a CONFIRMED accounting record (non-streaming path).

    This runs AFTER the provider responds but BEFORE the HTTP response
    is sent to the client. If this fails, raise so the gateway returns 503.
    """
    try:
        event_id = await write_confirmed(ctx, session=session)
        ctx.accounting_event_id = event_id
        ctx.accounting_status = "CONFIRMED"
    except AccountingError as exc:
        log.error(
            "accounting_failed_returning_503",
            request_id=str(ctx.request_id),
            error=str(exc),
        )
        raise COALAIError(
            COALAIErrorType.ACCOUNTING_WRITE_FAILED,
            "Accounting write failed; request cannot be processed",
            http_status=503,
        ) from exc


async def run_accounting_pending(
    ctx: PipelineContext,
    estimated_input_tokens: int,
    session: AsyncSession,
) -> uuid.UUID:
    """
    Write a PENDING accounting record before streaming begins.

    If this fails, raise so the gateway returns 503 before any chunk is sent.
    Returns the event_id needed for the Phase 2 confirm call.
    """
    try:
        event_id = await write_pending(
            ctx,
            estimated_input_tokens=estimated_input_tokens,
            session=session,
        )
        ctx.accounting_event_id = event_id
        ctx.accounting_status = "PENDING"
        return event_id
    except AccountingError as exc:
        log.error(
            "accounting_pending_failed_returning_503",
            request_id=str(ctx.request_id),
            error=str(exc),
        )
        raise COALAIError(
            COALAIErrorType.ACCOUNTING_WRITE_FAILED,
            "Accounting write failed; request cannot be processed",
            http_status=503,
        ) from exc
