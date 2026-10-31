"""
Fallback Routing Engine.

Orchestrates multi-provider/model fallback execution with circuit breaker protection.
Dispatches requests down an ordered chain of RouteTargets:
  - If a target's circuit breaker is OPEN, it is skipped fast.
  - If a target fails with retryable errors (after retry exhaustion), its breaker records the failure
    and the router transparently falls back to the next target.
  - If a target fails with a non-retryable error (e.g., 400 Bad Request), it fails immediately
    without trying fallbacks.
  - For streaming, fallback is permitted ONLY before the first chunk is emitted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from coalai.models.contracts import (
    ExecutionResult,
    NormalizedChunk,
    NormalizedRequest,
    PipelineContext,
    utcnow,
)
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import get_logger
from coalai.providers.registry import ProviderRegistry, RouteTarget
from coalai.reliability.circuit_breaker import CircuitBreaker
from coalai.reliability.retry import RetryPolicy, retry_with_backoff

log = get_logger(__name__)


class _TrackedStream:
    """Track a selected provider stream until it completes or is abandoned."""

    def __init__(
        self,
        first_chunk: NormalizedChunk | None,
        source: AsyncIterator[NormalizedChunk],
        breaker: CircuitBreaker,
        key: str,
        ticket: int,
        *,
        complete: bool = False,
    ) -> None:
        self._first_chunk = first_chunk
        self._source = source
        self._breaker = breaker
        self._key = key
        self._ticket = ticket
        self._closed = complete

    def __aiter__(self) -> _TrackedStream:
        return self

    async def __anext__(self) -> NormalizedChunk:
        if self._closed:
            raise StopAsyncIteration
        if self._first_chunk is not None:
            chunk = self._first_chunk
            self._first_chunk = None
            return chunk

        try:
            return await self._source.__anext__()
        except StopAsyncIteration:
            self._closed = True
            self._breaker.record_success(self._key, self._ticket)
            raise
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except COALAIError as exc:
            self._closed = True
            if exc.retryable:
                self._breaker.record_failure(self._key, self._ticket, exc)
            else:
                self._breaker.release(self._key, self._ticket)
            await self._close_source()
            raise
        except Exception:
            await self.aclose()
            raise

    async def _close_source(self) -> None:
        close = getattr(self._source, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                log.debug("provider_stream_close_failed", target=self._key, exc_info=True)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._breaker.release(self._key, self._ticket)
        await self._close_source()


class FallbackRouter:
    """Dispatches calls across an ordered target chain with circuit breaker guards."""

    def __init__(
        self,
        registry: ProviderRegistry,
        circuit_breaker: CircuitBreaker,
        retry_policy: RetryPolicy,
    ) -> None:
        self.registry = registry
        self.circuit_breaker = circuit_breaker
        self.retry_policy = retry_policy

    async def execute(
        self,
        ctx: PipelineContext,
        targets: list[RouteTarget],
        *,
        timeout_seconds: float,
    ) -> tuple[ExecutionResult, RouteTarget]:
        """
        Execute a non-streaming completion against the first available target.
        Returns the ExecutionResult and the target that succeeded.
        """
        if not targets:
            raise COALAIError(
                COALAIErrorType.ALL_PROVIDERS_FAILED,
                "No route targets configured for this request",
                http_status=503,
            )

        last_error: COALAIError | None = None

        for idx, target in enumerate(targets):
            key = target.key

            try:
                async with self.circuit_breaker.guard(key) as permitted:
                    if not permitted:
                        log.warning(
                            "circuit_breaker_open_skipping_target",
                            target=key,
                            request_id=str(ctx.request_id),
                        )
                        last_error = COALAIError(
                            COALAIErrorType.CIRCUIT_BREAKER_OPEN,
                            f"Circuit breaker is OPEN for target {key}",
                        )
                        continue

                    provider = self.registry.get_provider(target.provider_id)
                    if provider is None:
                        log.error("provider_not_found_in_registry", provider_id=target.provider_id)
                        continue

                    normalized_req = NormalizedRequest(
                        messages=ctx.messages,
                        model=target.model,
                        stream=False,
                        request_id=ctx.request_id,
                        max_tokens=ctx.max_tokens,
                        temperature=ctx.temperature,
                        stop=ctx.stop,
                        seed=ctx.seed,
                    )

                    log.debug("attempting_target_execution", target=key, request_id=str(ctx.request_id))

                    async def _call():
                        return await provider.complete(normalized_req, timeout_seconds=timeout_seconds)

                    result = await retry_with_backoff(
                        _call,
                        policy=self.retry_policy,
                        request_id=str(ctx.request_id),
                    )

                    return result, target

            except COALAIError as exc:
                last_error = exc
                if not exc.retryable:
                    log.debug("fallback_aborted_non_retryable", target=key, error_type=exc.error_type)
                    raise

                log.warning(
                    "target_failed_falling_back",
                    failed_target=key,
                    error_type=exc.error_type,
                    remaining_targets=len(targets) - (idx + 1),
                    request_id=str(ctx.request_id),
                )

        raise COALAIError(
            COALAIErrorType.ALL_PROVIDERS_FAILED,
            f"All provider targets failed or circuit breakers are OPEN. Last error: {last_error.message if last_error else 'none'}",
            http_status=503,
        )

    async def stream(
        self,
        ctx: PipelineContext,
        targets: list[RouteTarget],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[NormalizedChunk]:
        """
        Execute streaming with fallback allowed strictly before the first chunk is emitted.
        Once the first chunk is emitted, streaming is committed to that target.
        """
        if not targets:
            raise COALAIError(
                COALAIErrorType.ALL_PROVIDERS_FAILED,
                "No route targets configured for streaming",
                http_status=503,
            )

        last_error: COALAIError | None = None

        for idx, target in enumerate(targets):
            key = target.key

            ticket = self.circuit_breaker.can_execute(key)
            if ticket is None:
                log.warning("circuit_breaker_open_skipping_stream_target", target=key)
                last_error = COALAIError(
                    COALAIErrorType.CIRCUIT_BREAKER_OPEN,
                    f"Circuit breaker is OPEN for target {key}",
                )
                continue

            provider = self.registry.get_provider(target.provider_id)
            if provider is None:
                self.circuit_breaker.release(key, ticket)
                continue

            normalized_req = NormalizedRequest(
                messages=ctx.messages,
                model=target.model,
                stream=True,
                request_id=ctx.request_id,
                max_tokens=ctx.max_tokens,
                temperature=ctx.temperature,
                stop=ctx.stop,
                seed=ctx.seed,
            )

            try:
                try:
                    chunk_iter = await provider.stream(normalized_req, timeout_seconds=timeout_seconds)
                    first_chunk = await chunk_iter.__anext__()
                    ctx.selected_provider = target.provider_id
                    ctx.selected_model = target.model
                    if first_chunk.content and ctx.first_token_at is None:
                        ctx.first_token_at = utcnow()
                    return _TrackedStream(first_chunk, chunk_iter, self.circuit_breaker, key, ticket)

                except StopAsyncIteration:
                    self.circuit_breaker.record_success(key, ticket)
                    ctx.selected_provider = target.provider_id
                    ctx.selected_model = target.model
                    return _TrackedStream(
                        None,
                        _empty_stream(),
                        self.circuit_breaker,
                        key,
                        ticket,
                        complete=True,
                    )

                except asyncio.CancelledError:
                    self.circuit_breaker.release(key, ticket)
                    raise

                except COALAIError as exc:
                    if not exc.retryable:
                        self.circuit_breaker.release(key, ticket)
                    else:
                        self.circuit_breaker.record_failure(key, ticket, exc)
                    raise

                except Exception:
                    self.circuit_breaker.release(key, ticket)
                    raise

            except COALAIError as exc:
                last_error = exc
                if not exc.retryable:
                    raise

                log.warning(
                    "stream_target_init_failed_falling_back",
                    failed_target=key,
                    error_type=exc.error_type,
                    remaining_targets=len(targets) - (idx + 1),
                )

        raise COALAIError(
            COALAIErrorType.ALL_PROVIDERS_FAILED,
            f"All streaming provider targets failed. Last error: {last_error.message if last_error else 'none'}",
            http_status=503,
        )


async def _empty_stream() -> AsyncIterator[NormalizedChunk]:
    if False:
        yield NormalizedChunk(content="", finish_reason=None)
