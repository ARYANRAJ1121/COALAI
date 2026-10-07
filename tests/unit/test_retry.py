"""
Unit tests for the retry policy.

Tests:
  - Retryable error → retries up to max_attempts
  - Non-retryable error → does NOT retry (raises immediately)
  - Successful call after retry → returns result
  - max_attempts respected → raises after N attempts
  - Jitter is applied → retry delays have variance (not constant)
  - Retry-After header honoured → uses provider delay (capped)
  - Non-COALAIError → re-raised immediately (no retry)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, call

import pytest

from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.reliability.retry import RetryPolicy, retry_with_backoff


def timeout_error() -> COALAIError:
    return COALAIError(COALAIErrorType.PROVIDER_TIMEOUT, "timed out")


def bad_request_error() -> COALAIError:
    return COALAIError(COALAIErrorType.PROVIDER_BAD_REQUEST, "bad request")


def rate_limited_error(retry_after: float = 0.0) -> COALAIError:
    return COALAIError(
        COALAIErrorType.PROVIDER_RATE_LIMITED,
        "rate limited",
        retry_after_seconds=retry_after,
    )


FAST_POLICY = RetryPolicy(
    max_attempts=3,
    initial_delay_seconds=0.01,
    backoff_factor=2.0,
    jitter_factor=0.1,
    max_delay_seconds=1.0,
)


# ── Retryable error retries ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_retryable_error() -> None:
    """Retryable error triggers retries up to max_attempts."""
    call_count = 0

    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise timeout_error()
        return "success"

    result = await retry_with_backoff(fn, policy=FAST_POLICY)
    assert result == "success"
    assert call_count == 3


# ── Non-retryable error does not retry ────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_retry_on_non_retryable() -> None:
    """Non-retryable error raises immediately without retry."""
    call_count = 0

    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        raise bad_request_error()

    with pytest.raises(COALAIError) as exc_info:
        await retry_with_backoff(fn, policy=FAST_POLICY)

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_BAD_REQUEST
    assert call_count == 1  # Called exactly once — no retry


# ── Max attempts respected ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_attempts_respected() -> None:
    """After max_attempts, raises the last error."""
    call_count = 0

    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        raise timeout_error()

    with pytest.raises(COALAIError) as exc_info:
        await retry_with_backoff(fn, policy=FAST_POLICY)

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_TIMEOUT
    assert call_count == FAST_POLICY.max_attempts


# ── First attempt succeeds ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_attempt_success() -> None:
    """If first call succeeds, no retry occurs."""
    async def fn() -> str:
        return "hello"

    result = await retry_with_backoff(fn, policy=FAST_POLICY)
    assert result == "hello"


# ── Jitter produces variance ───────────────────────────────────────────────────


def test_retry_policy_delay_has_jitter() -> None:
    """Computed delays for the same attempt must vary (jitter is random)."""
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay_seconds=1.0,
        backoff_factor=2.0,
        jitter_factor=0.5,
        max_delay_seconds=30.0,
    )
    # Sample 20 delays for attempt 1
    delays = [policy.compute_delay(1) for _ in range(20)]
    # With 50% jitter, delays should not all be the same
    assert len(set(round(d, 4) for d in delays)) > 1, \
        "All delays were identical — jitter is not being applied"
    # All delays should be within the jitter band [0.5, 1.5] for initial=1.0, jitter=0.5
    for d in delays:
        assert 0.4 <= d <= 1.6, f"Delay {d} is outside expected jitter band"


# ── Retry-After respected ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_after_respected() -> None:
    """When provider returns Retry-After, that delay is used (not backoff)."""
    delays_observed: list[float] = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay: float) -> None:
        delays_observed.append(delay)

    call_count = 0

    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise rate_limited_error(retry_after=0.05)
        return "ok"

    policy = RetryPolicy(
        max_attempts=3,
        initial_delay_seconds=10.0,  # Would be used if Retry-After is ignored
        backoff_factor=2.0,
        jitter_factor=0.0,
        max_delay_seconds=30.0,
    )

    import coalai.reliability.retry as retry_module

    original = retry_module.asyncio.sleep  # type: ignore
    retry_module.asyncio.sleep = mock_sleep  # type: ignore
    try:
        result = await retry_with_backoff(fn, policy=policy)
    finally:
        retry_module.asyncio.sleep = original  # type: ignore

    assert result == "ok"
    assert len(delays_observed) == 1
    # Delay should be Retry-After (0.05), NOT the backoff (10.0)
    assert delays_observed[0] <= 0.1, f"Retry-After not respected, delay was {delays_observed[0]}"


# ── Non-COALAIError is re-raised immediately ───────────────────────────────────


@pytest.mark.asyncio
async def test_non_coalai_error_not_retried() -> None:
    """A non-COALAIError (e.g., ValueError) is re-raised immediately."""
    call_count = 0

    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("programming bug")

    with pytest.raises(ValueError, match="programming bug"):
        await retry_with_backoff(fn, policy=FAST_POLICY)

    assert call_count == 1
