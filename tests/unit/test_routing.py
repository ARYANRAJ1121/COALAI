"""
Unit tests for ProviderRegistry and FallbackRouter.
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
import pytest

from coalai.models.contracts import (
    ExecutionResult,
    FinishReason,
    Message,
    NormalizedChunk,
    NormalizedRequest,
    PipelineContext,
    utcnow,
)
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.providers.base import LLMProvider, ProviderHealth, ProviderStatus
from coalai.providers.registry import ProviderRegistry, RouteTarget
from coalai.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from coalai.reliability.retry import RetryPolicy
from coalai.reliability.routing import FallbackRouter


class MockProvider:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.complete_mock = AsyncMock()
        self.stream_mock = AsyncMock()

    async def complete(self, request: NormalizedRequest, *, timeout_seconds: float) -> ExecutionResult:
        return await self.complete_mock(request, timeout_seconds=timeout_seconds)

    async def stream(self, request: NormalizedRequest, *, timeout_seconds: float) -> AsyncIterator[NormalizedChunk]:
        return await self.stream_mock(request, timeout_seconds=timeout_seconds)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_id=self.provider_id, status=ProviderStatus.HEALTHY, latency_ms=10.0)


@pytest.fixture
def sample_context() -> PipelineContext:
    return PipelineContext(
        request_id=uuid.uuid4(),
        trace_id="test-trace",
        received_at=utcnow(),
        messages=[Message(role="user", content="Hello")],
        model_hint="llama3",
        stream=False,
    )


@pytest.fixture
def fast_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=2,
        initial_delay_seconds=0.01,
        backoff_factor=1.0,
        jitter_factor=0.0,
        max_delay_seconds=0.1,
    )


@pytest.fixture
def router_setup(fast_retry_policy: RetryPolicy):
    registry = ProviderRegistry()
    cb = CircuitBreaker(
        config=CircuitBreakerConfig(failure_threshold=2, recovery_time_seconds=0.1)
    )
    router = FallbackRouter(
        registry=registry,
        circuit_breaker=cb,
        retry_policy=fast_retry_policy,
    )
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    registry.register_provider(p1)
    registry.register_provider(p2)

    return router, registry, cb, p1, p2


@pytest.mark.asyncio
async def test_successful_primary_target(router_setup, sample_context: PipelineContext):
    router, registry, cb, p1, p2 = router_setup
    now = utcnow()

    expected_result = ExecutionResult(
        provider_used="p1",
        model_used="model-a",
        content="Primary response",
        finish_reason=FinishReason.STOP,
        input_tokens=10,
        output_tokens=20,
        provider_request_id="req-1",
        execution_started_at=now,
        execution_completed_at=now,
        success=True,
    )
    p1.complete_mock.return_value = expected_result

    targets = [RouteTarget("p1", "model-a"), RouteTarget("p2", "model-b")]
    result, target_used = await router.execute(sample_context, targets, timeout_seconds=5.0)

    assert result.content == "Primary response"
    assert target_used.provider_id == "p1"
    assert p1.complete_mock.await_count == 1
    assert p2.complete_mock.await_count == 0


@pytest.mark.asyncio
async def test_fallback_on_primary_failure(router_setup, sample_context: PipelineContext):
    router, registry, cb, p1, p2 = router_setup
    now = utcnow()

    # Primary fails with retryable error (e.g. 503)
    p1.complete_mock.side_effect = COALAIError(COALAIErrorType.PROVIDER_SERVER_ERROR, "Primary down")

    # Secondary succeeds
    fallback_result = ExecutionResult(
        provider_used="p2",
        model_used="model-b",
        content="Fallback response",
        finish_reason=FinishReason.STOP,
        input_tokens=10,
        output_tokens=20,
        provider_request_id="req-2",
        execution_started_at=now,
        execution_completed_at=now,
        success=True,
    )
    p2.complete_mock.return_value = fallback_result

    targets = [RouteTarget("p1", "model-a"), RouteTarget("p2", "model-b")]
    result, target_used = await router.execute(sample_context, targets, timeout_seconds=5.0)

    assert result.content == "Fallback response"
    assert target_used.provider_id == "p2"
    assert p1.complete_mock.await_count == 2  # Retried 2 times according to fast_retry_policy
    assert p2.complete_mock.await_count == 1


@pytest.mark.asyncio
async def test_skips_open_circuit_breaker(router_setup, sample_context: PipelineContext):
    router, registry, cb, p1, p2 = router_setup
    now = utcnow()

    # Pre-trip primary breaker
    cb.trip("p1:model-a")

    fallback_result = ExecutionResult(
        provider_used="p2",
        model_used="model-b",
        content="Fallback response",
        finish_reason=FinishReason.STOP,
        input_tokens=10,
        output_tokens=20,
        provider_request_id="req-3",
        execution_started_at=now,
        execution_completed_at=now,
        success=True,
    )
    p2.complete_mock.return_value = fallback_result

    targets = [RouteTarget("p1", "model-a"), RouteTarget("p2", "model-b")]
    result, target_used = await router.execute(sample_context, targets, timeout_seconds=5.0)

    assert result.content == "Fallback response"
    assert target_used.provider_id == "p2"
    # p1 should not even be called because breaker was OPEN!
    assert p1.complete_mock.await_count == 0
    assert p2.complete_mock.await_count == 1


@pytest.mark.asyncio
async def test_non_retryable_error_does_not_fallback(router_setup, sample_context: PipelineContext):
    router, registry, cb, p1, p2 = router_setup

    # Primary fails with client error (400 Bad Request)
    p1.complete_mock.side_effect = COALAIError(
        COALAIErrorType.PROVIDER_BAD_REQUEST, "Malformed prompt", retryable=False
    )

    targets = [RouteTarget("p1", "model-a"), RouteTarget("p2", "model-b")]

    with pytest.raises(COALAIError) as exc_info:
        await router.execute(sample_context, targets, timeout_seconds=5.0)

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_BAD_REQUEST
    assert p1.complete_mock.await_count == 1
    assert p2.complete_mock.await_count == 0  # Did not attempt fallback!


@pytest.mark.asyncio
async def test_all_targets_fail_raises_all_providers_failed(router_setup, sample_context: PipelineContext):
    router, registry, cb, p1, p2 = router_setup

    p1.complete_mock.side_effect = COALAIError(COALAIErrorType.PROVIDER_TIMEOUT, "Timeout 1")
    p2.complete_mock.side_effect = COALAIError(COALAIErrorType.PROVIDER_TIMEOUT, "Timeout 2")

    targets = [RouteTarget("p1", "model-a"), RouteTarget("p2", "model-b")]

    with pytest.raises(COALAIError) as exc_info:
        await router.execute(sample_context, targets, timeout_seconds=5.0)

    assert exc_info.value.error_type == COALAIErrorType.ALL_PROVIDERS_FAILED
