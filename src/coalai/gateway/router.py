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

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from coalai.accounting.service import confirm_pending
from coalai.config import settings
from coalai.db.session import get_db_session
from coalai.gateway import pipeline
from coalai.gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
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
from coalai.models.contracts import PipelineContext, FinishReason, utcnow
from coalai.models.errors import COALAIError
from coalai.observability.logging import clear_request_context, get_logger
from coalai.providers.ollama import OllamaProvider
from coalai.reliability.retry import RetryPolicy

log = get_logger(__name__)

router = APIRouter()

_BEARER_PREFIX = "Bearer "


def _extract_bearer_token(request: Request) -> str:
    """Extract bearer token from Authorization header. Raises 401 on failure."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith(_BEARER_PREFIX):
        raise COALAIError(
            __import__("coalai.models.errors", fromlist=["COALAIErrorType"]).COALAIErrorType.AUTHENTICATION_FAILED,
            "Missing or malformed Authorization header. Expected: Bearer <key>",
            http_status=401,
        )
    return auth_header[len(_BEARER_PREFIX):]


def _coalai_error_to_json_response(exc: COALAIError, request_id: str | None) -> JSONResponse:
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


def _get_provider(request: Request) -> OllamaProvider:
    """Retrieve the OllamaProvider singleton from app state."""
    return request.app.state.ollama_provider


def _get_retry_policy(request: Request) -> RetryPolicy:
    return request.app.state.retry_policy


@router.get("/health", response_model=HealthResponse, tags=["observability"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 if the process is alive."""
    return HealthResponse(version=settings.app_version)


@router.get("/ready", response_model=ReadinessResponse, tags=["observability"])
async def ready(session: AsyncSession = Depends(get_db_session)) -> ReadinessResponse:
    """
    Readiness probe. Checks PostgreSQL and Redis connectivity.
    Returns 200 only if both are reachable.
    """
    from coalai.cache.redis_client import redis_ping

    postgres_ok = False
    redis_ok = False

    try:
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        pass

    redis_ok = await redis_ping()

    status_str = "ready" if (postgres_ok and redis_ok) else "not_ready"
    http_status = 200 if (postgres_ok and redis_ok) else 503

    return JSONResponse(
        status_code=http_status,
        content=ReadinessResponse(
            status=status_str,
            postgres=postgres_ok,
            redis=redis_ok,
        ).model_dump(),
    )


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
    raw_key: str | None = None
    ctx: PipelineContext | None = None

    try:
        # Extract bearer token (before building context — no request_id yet)
        raw_key = _extract_bearer_token(request)

        # Stage 1: Build pipeline context
        ctx = pipeline.build_context(req)

        # Stage 2: Auth
        await pipeline.run_auth(ctx, raw_key, session)

        # Stage 3: Rate limiting
        await pipeline.run_rate_limit(ctx)

        # Stage 4: Resolve model
        model = pipeline.resolve_model(ctx)

        provider = _get_provider(request)
        retry_policy = _get_retry_policy(request)

        if not req.stream:
            return await _handle_buffered(ctx, model, provider, retry_policy, session)
        else:
            return await _handle_streaming(ctx, model, provider, session)

    except COALAIError as exc:
        log.warning(
            "request_failed",
            error_type=exc.error_type,
            http_status=exc.http_status,
            request_id=str(ctx.request_id) if ctx else None,
        )
        return _coalai_error_to_json_response(exc, str(ctx.request_id) if ctx else None)

    except Exception as exc:
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
        if ctx:
            ctx.responded_at = utcnow()
            log.info(
                "request_completed",
                http_status=ctx.http_status,
                total_latency_ms=round(ctx.total_latency_ms, 1),
            )
        clear_request_context()


async def _handle_buffered(
    ctx: PipelineContext,
    model: str,
    provider: OllamaProvider,
    retry_policy: RetryPolicy,
    session: AsyncSession,
) -> JSONResponse:
    """Handle a non-streaming chat completion."""
    # Stage 5: Execute
    await pipeline.execute_completion(ctx, model, provider, retry_policy)
    assert ctx.result is not None

    # Stage 6: Account (BEFORE returning to client — v0.3 Option A)
    await pipeline.run_accounting_confirmed(ctx, session)
    await session.commit()

    ctx.http_status = 200

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
            fallback_triggered=False,
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
    provider: OllamaProvider,
    session: AsyncSession,
) -> StreamingResponse:
    """
    Handle a streaming chat completion.

    Accounting phase 1 (PENDING write) happens BEFORE any chunk is sent.
    Accounting phase 2 (CONFIRMED update) happens after the stream ends.
    """
    # Estimate input tokens for the PENDING accounting write.
    # Simple character-count heuristic; accurate counts come from the stream's final chunk.
    estimated_input_tokens = sum(len(m.content) for m in ctx.messages) // 4

    # Write PENDING accounting record BEFORE any chunk is sent.
    # If this fails → raises COALAIError → returns 503 (caught in chat_completions)
    event_id = await pipeline.run_accounting_pending(ctx, estimated_input_tokens, session)
    await session.commit()

    # Get the stream iterator
    stream_iter = await pipeline.execute_stream(ctx, model, provider)

    async def _sse_generator() -> AsyncIterator[str]:
        """Yield SSE-formatted chunks, then confirm accounting after stream ends."""
        actual_input_tokens = 0
        actual_output_tokens = 0

        try:
            async for chunk in await stream_iter:
                if chunk.content:
                    sse_chunk = ChatCompletionChunk(
                        id=f"chatcmpl-{ctx.request_id}",
                        model=model,
                        choices=[
                            StreamingChoice(
                                delta=StreamingDelta(content=chunk.content),
                                finish_reason=chunk.finish_reason.value if chunk.finish_reason else None,
                            )
                        ],
                    )
                    yield f"data: {sse_chunk.model_dump_json()}\n\n"

                # Update token counts from final chunk
                if chunk.input_tokens is not None:
                    actual_input_tokens = chunk.input_tokens
                if chunk.output_tokens is not None:
                    actual_output_tokens = chunk.output_tokens

            yield "data: [DONE]\n\n"

        finally:
            # Phase 2: confirm accounting after stream completes or errors
            # This runs even on client disconnect (via generator cleanup)
            from coalai.db.session import get_session

            async with get_session() as confirm_session:
                await confirm_pending(
                    event_id,
                    actual_input_tokens=actual_input_tokens or estimated_input_tokens,
                    actual_output_tokens=actual_output_tokens,
                    estimated_cost_usd=0.0,
                    session=confirm_session,
                    request_id=str(ctx.request_id),
                )
            ctx.http_status = 200

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-COALAI-Request-ID": str(ctx.request_id),
        },
    )
