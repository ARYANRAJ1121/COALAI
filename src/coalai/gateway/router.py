"""
FastAPI route handlers for the COALAI gateway.

This module contains ONLY:
  - HTTP request parsing
  - Pipeline orchestration (calling pipeline.py functions in order)
  - HTTP response serialization
  - Error → HTTP response translation

No business logic lives here. Each route handler is a thin shell
around the pipeline functions.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from coalai.accounting.service import confirm_pending
from coalai.config import settings
from coalai.db.session import get_db_session, get_session
from coalai.gateway import pipeline
from coalai.gateway.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    COALAIResponseExtension,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    StreamingChoice,
    StreamingDelta,
    UsageInfo,
)
from coalai.models.contracts import PipelineContext, utcnow
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from coalai.observability.metrics import (
    active_requests,
    get_metrics_response,
    record_accounting_failure,
    record_request_complete,
)
from coalai.providers.ollama import OllamaProvider
from coalai.providers.registry import ProviderRegistry, RouteTarget
from coalai.reliability.retry import RetryPolicy
from coalai.reliability.routing import FallbackRouter

log = get_logger(__name__)

router = APIRouter()

_BEARER_PREFIX = "Bearer "


def _extract_bearer_token(request: Request) -> str:
    """Extract bearer token from Authorization header. Raises COALAIError on failure."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith(_BEARER_PREFIX):
        raise COALAIError(
            COALAIErrorType.AUTHENTICATION_FAILED,
            "Missing or malformed Authorization header. Expected: Bearer <key>",
            http_status=401,
        )
    return auth_header[len(_BEARER_PREFIX):]


def _error_response(exc: COALAIError, request_id: str | None) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content=ErrorResponse(
            error=ErrorDetail(
                error_type=exc.error_type,
                message=exc.message,
                request_id=request_id,
            )
        ).model_dump(),
    )


def _finalize_request(ctx: PipelineContext, *, status: str | None = None) -> None:
    """Record request completion when the response body has actually finished."""
    ctx.responded_at = utcnow()
    final_status = status or (
        "success" if ctx.http_status == 200 else str(getattr(ctx, "_last_error_type", "error"))
    )
    try:
        record_request_complete(ctx, status=final_status)
        log.info(
            "request_completed",
            http_status=ctx.http_status,
            outcome=final_status,
            total_latency_ms=round(ctx.total_latency_ms, 1),
        )
    finally:
        active_requests.dec()
        clear_request_context()


def _get_provider(request: Request) -> OllamaProvider:
    """Retrieve the OllamaProvider singleton from app state."""
    return request.app.state.ollama_provider  # type: ignore[no-any-return]


def _get_retry_policy(request: Request) -> RetryPolicy:
    return request.app.state.retry_policy  # type: ignore[no-any-return]


def _get_fallback_router(request: Request) -> FallbackRouter | None:
    return getattr(request.app.state, "fallback_router", None)


def _get_provider_registry(request: Request) -> ProviderRegistry | None:
    return getattr(request.app.state, "provider_registry", None)


# ── Liveness & readiness ───────────────────────────────────────────────────────


@router.get("/metrics", tags=["observability"], include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)


@router.get("/health", response_model=HealthResponse, tags=["observability"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 if the process is alive."""
    return HealthResponse(version=settings.app_version)


@router.get("/ready", tags=["observability"])
async def ready(session: AsyncSession = Depends(get_db_session)) -> JSONResponse:
    """
    Readiness probe. Checks PostgreSQL and Redis connectivity.
    Returns 200 only when both are reachable.
    """
    from sqlalchemy import text

    from coalai.cache.redis_client import redis_ping

    postgres_ok = False
    redis_ok = False

    try:
        await session.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        pass

    redis_ok = await redis_ping()

    http_status = 200 if (postgres_ok and redis_ok) else 503
    return JSONResponse(
        status_code=http_status,
        content=ReadinessResponse(
            status="ready" if (postgres_ok and redis_ok) else "not_ready",
            postgres=postgres_ok,
            redis=redis_ok,
        ).model_dump(),
    )


# ── Chat completions ───────────────────────────────────────────────────────────


@router.post("/v1/chat/completions", tags=["completions"])
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> JSONResponse | StreamingResponse:
    """
    OpenAI-compatible chat completion endpoint.

    Supports both buffered (stream=false) and streaming (stream=true) modes.
    Returns structured JSON errors on all failure cases.
    """
    ctx: PipelineContext | None = None
    stream_lifecycle_deferred = False
    active_requests.inc()
    try:
        # Extract bearer token (pre-auth — tenant_id NOT in logs yet)
        raw_key = _extract_bearer_token(request)

        # Stage 1: Build pipeline context
        ctx = pipeline.build_context(req)

        # Stage 2: Auth — after this, tenant_id is bound to log context
        await pipeline.run_auth(ctx, raw_key, session)

        # Stage 3: Rate limiting (fail-open if Redis unavailable)
        await pipeline.run_rate_limit(ctx)

        # Stage 4: Resolve model & targets
        model = pipeline.resolve_model(ctx)
        registry = _get_provider_registry(request)
        fallback_router = _get_fallback_router(request)

        if registry is not None:
            targets = registry.resolve_targets(ctx.model_hint)
        else:
            targets = [RouteTarget(provider_id="ollama", model=model)]

        provider = _get_provider(request)
        retry_policy = _get_retry_policy(request)

        if not req.stream:
            return await _handle_buffered(
                ctx=ctx,
                model=model,
                targets=targets,
                provider=provider,
                retry_policy=retry_policy,
                router=fallback_router,
                session=session,
            )
        else:
            streaming_response = await _handle_streaming(
                ctx=ctx,
                model=model,
                targets=targets,
                provider=provider,
                router=fallback_router,
                session=session,
            )
            stream_lifecycle_deferred = True
            return streaming_response

    except COALAIError as exc:
        if ctx:
            ctx.http_status = exc.http_status
            ctx._last_error_type = str(exc.error_type)
        log.warning(
            "request_failed",
            error_type=str(exc.error_type),
            http_status=exc.http_status,
            request_id=str(ctx.request_id) if ctx else None,
        )
        return _error_response(exc, str(ctx.request_id) if ctx else None)

    except Exception as exc:
        if ctx:
            ctx.http_status = 500
            ctx._last_error_type = "internal_error"
        log.exception(
            "unexpected_error",
            request_id=str(ctx.request_id) if ctx else None,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    error_type="internal_error",
                    message="An unexpected error occurred.",
                    request_id=str(ctx.request_id) if ctx else None,
                )
            ).model_dump(),
        )
    finally:
        if not stream_lifecycle_deferred:
            if ctx:
                _finalize_request(ctx)
            else:
                active_requests.dec()
                clear_request_context()
        else:
            # Streaming runs after this handler returns; the body generator
            # rebinds the request context when it starts.
            clear_request_context()


async def _handle_buffered(
    ctx: PipelineContext,
    model: str,
    targets: list[RouteTarget],
    provider: OllamaProvider,
    retry_policy: RetryPolicy,
    router: FallbackRouter | None,
    session: AsyncSession,
) -> JSONResponse:
    """
    Non-streaming completion.
    Accounting write (CONFIRMED) happens BEFORE HTTP 200 is returned — v0.3 Option A.
    """
    # Stage 5: Execute with circuit breaker & fallback router
    await pipeline.execute_completion(
        ctx,
        model=model,
        provider=provider,
        policy=retry_policy,
        targets=targets,
        router=router,
    )
    assert ctx.result is not None

    # Stage 6: Account — must succeed before responding
    await pipeline.run_accounting_confirmed(ctx, session)
    await session.commit()

    ctx.http_status = 200

    fallback_triggered = bool(
        targets
        and (
            ctx.result.provider_used != targets[0].provider_id
            or ctx.result.model_used != targets[0].model
        )
    )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{ctx.request_id}",
        model=ctx.result.model_used,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionResponseMessage(content=ctx.result.content or ""),
                finish_reason=ctx.result.finish_reason.value,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=ctx.result.input_tokens,
            completion_tokens=ctx.result.output_tokens,
            total_tokens=ctx.result.total_tokens,
        ),
        coalai=COALAIResponseExtension(
            request_id=str(ctx.request_id),
            trace_id=ctx.trace_id,
            provider_used=ctx.result.provider_used,
            model_used=ctx.result.model_used,
            fallback_triggered=fallback_triggered,
            attempt_number=ctx.result.attempt_number,
            cache_hit=False,
            cache_type=None,
            estimated_cost_usd=ctx.estimated_cost_usd,
            gateway_latency_ms=round(ctx.total_latency_ms - ctx.provider_latency_ms, 1),
            provider_latency_ms=round(ctx.provider_latency_ms, 1),
            total_latency_ms=round(ctx.total_latency_ms, 1),
            ttft_ms=ctx.result.ttft_ms,
        ),
    )
    return JSONResponse(content=response.model_dump(), status_code=200)


async def _handle_streaming(
    ctx: PipelineContext,
    model: str,
    targets: list[RouteTarget],
    provider: OllamaProvider,
    router: FallbackRouter | None,
    session: AsyncSession,
) -> StreamingResponse:
    """
    Streaming completion via SSE.

    v0.3 Option A — two-phase accounting:
      Phase 1: INSERT PENDING before first chunk (raises 503 if it fails).
      Phase 2: UPDATE to CONFIRMED after stream ends (reconciliation handles failures).
    """
    # Rough input-token estimate for the PENDING write (char count ÷ 4).
    estimated_input_tokens = sum(len(m.content) for m in ctx.messages) // 4

    # Open the selected route before writing the pending row so fallback
    # metadata is known and an unavailable provider cannot leave a ghost row.
    stream_iter = await pipeline.execute_stream(
        ctx,
        model=model,
        provider=provider,
        targets=targets,
        router=router,
    )
    try:
        # The provider is selected, but no chunk has been sent to the client.
        event_id = await pipeline.run_accounting_pending(ctx, estimated_input_tokens, session)
        await session.commit()
    except BaseException:
        close = getattr(stream_iter, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception as exc:  # noqa: BLE001 - preserve the accounting failure/cancellation
                log.warning("provider_stream_close_failed", request_id=str(ctx.request_id), error=str(exc))
        ctx.first_token_at = None  # The prefetched chunk was never delivered to the client.
        raise

    selected_model = ctx.selected_model or model
    ctx.http_status = 200

    async def _sse_generator() -> AsyncGenerator[str, None]:
        actual_input_tokens = 0
        actual_output_tokens = 0
        outcome = "success"
        if ctx.tenant_context is not None:
            bind_request_context(
                request_id=str(ctx.request_id),
                trace_id=ctx.trace_id,
                tenant_id=str(ctx.tenant_context.tenant_id),
            )

        try:
            async for chunk in stream_iter:
                if chunk.content:
                    if ctx.first_token_at is None:
                        ctx.first_token_at = utcnow()
                    sse_chunk = ChatCompletionChunk(
                        id=f"chatcmpl-{ctx.request_id}",
                        model=selected_model,
                        choices=[
                            StreamingChoice(
                                delta=StreamingDelta(content=chunk.content),
                                finish_reason=(
                                    chunk.finish_reason.value if chunk.finish_reason else None
                                ),
                            )
                        ],
                    )
                    yield f"data: {sse_chunk.model_dump_json()}\n\n"

                if chunk.input_tokens is not None:
                    actual_input_tokens = chunk.input_tokens
                    ctx.stream_input_tokens = chunk.input_tokens
                if chunk.output_tokens is not None:
                    actual_output_tokens = chunk.output_tokens
                    ctx.stream_output_tokens = chunk.output_tokens

            ctx.provider_completed_at = utcnow()
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            outcome = "cancelled"
            ctx.provider_completed_at = utcnow()
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors are normalized at this boundary
            outcome = "stream_error"
            ctx.provider_completed_at = utcnow()
            log.error("stream_response_failed", request_id=str(ctx.request_id), error=str(exc))
            raise
        finally:
            close = getattr(stream_iter, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask stream outcome
                    log.warning("provider_stream_close_failed", request_id=str(ctx.request_id), error=str(exc))
            ctx.stream_input_tokens = actual_input_tokens or estimated_input_tokens
            ctx.stream_output_tokens = actual_output_tokens
            try:
                # Phase 2 uses a fresh session because the request session may be closed.
                async with get_session() as confirm_session:
                    await confirm_pending(
                        event_id,
                        actual_input_tokens=actual_input_tokens or estimated_input_tokens,
                        actual_output_tokens=actual_output_tokens,
                        estimated_cost_usd=0.0,
                        session=confirm_session,
                        request_id=str(ctx.request_id),
                    )
            except Exception as exc:
                record_accounting_failure(phase="confirm")
                log.error(
                    "accounting_confirmation_session_failed",
                    event_id=str(event_id),
                    request_id=str(ctx.request_id),
                    error=str(exc),
                )
            finally:
                _finalize_request(ctx, status=outcome)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-COALAI-Request-ID": str(ctx.request_id),
        },
    )
