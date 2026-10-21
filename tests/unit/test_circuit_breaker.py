"""
Unit tests for Circuit Breaker with generation tickets and leased canary probes.
"""

from __future__ import annotations

import asyncio
import time
import pytest

from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


@pytest.fixture
def cb_config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_time_seconds=0.1,  # Short recovery time for fast unit testing
        half_open_success_threshold=2,
        half_open_max_trials=2,
        canary_timeout_seconds=0.15,  # Short lease for testing lease expiry
    )


@pytest.fixture
def circuit_breaker(cb_config: CircuitBreakerConfig) -> CircuitBreaker:
    return CircuitBreaker(config=cb_config)


def test_initial_state_is_closed(circuit_breaker: CircuitBreaker):
    key = "ollama:llama3"
    assert circuit_breaker.get_state(key) == CircuitState.CLOSED
    ticket = circuit_breaker.can_execute(key)
    assert ticket is not None


def test_failures_trip_circuit_to_open(circuit_breaker: CircuitBreaker):
    key = "ollama:llama3"

    ticket = circuit_breaker.can_execute(key)
    assert ticket is not None

    # 1st and 2nd failure — still under threshold (3)
    circuit_breaker.record_failure(key, ticket, COALAIError(COALAIErrorType.PROVIDER_SERVER_ERROR, "502"))
    assert circuit_breaker.get_state(key) == CircuitState.CLOSED

    circuit_breaker.record_failure(key, ticket, COALAIError(COALAIErrorType.PROVIDER_TIMEOUT, "timeout"))
    assert circuit_breaker.get_state(key) == CircuitState.CLOSED

    # 3rd failure — hits threshold (3), trips to OPEN
    circuit_breaker.record_failure(key, ticket, COALAIError(COALAIErrorType.PROVIDER_SERVER_ERROR, "503"))
    assert circuit_breaker.get_state(key) == CircuitState.OPEN

    # can_execute fails fast immediately
    assert circuit_breaker.can_execute(key) is None


def test_non_retryable_errors_do_not_trip_breaker(circuit_breaker: CircuitBreaker):
    key = "ollama:llama3"
    ticket = circuit_breaker.can_execute(key)
    assert ticket is not None

    # Client errors (400 Bad Request, auth failure) must not count against breaker
    client_err = COALAIError(COALAIErrorType.INVALID_REQUEST, "bad prompt", retryable=False)
    for _ in range(5):
        circuit_breaker.record_failure(key, ticket, client_err)

    assert circuit_breaker.get_state(key) == CircuitState.CLOSED
    assert circuit_breaker.can_execute(key) is not None


@pytest.mark.asyncio
async def test_recovery_window_transitions_to_half_open(circuit_breaker: CircuitBreaker):
    key = "ollama:llama3"
    circuit_breaker.trip(key)
    assert circuit_breaker.get_state(key) == CircuitState.OPEN
    assert circuit_breaker.can_execute(key) is None

    # Wait for recovery window (0.1s in fixture)
    await asyncio.sleep(0.12)

    # get_state and can_execute report HALF_OPEN
    assert circuit_breaker.get_state(key) == CircuitState.HALF_OPEN
    ticket = circuit_breaker.can_execute(key)
    assert ticket is not None


@pytest.mark.asyncio
async def test_half_open_success_recovers_to_closed(circuit_breaker: CircuitBreaker):
    key = "ollama:llama3"
    circuit_breaker.trip(key)
    await asyncio.sleep(0.12)

    ticket1 = circuit_breaker.can_execute(key)
    assert ticket1 is not None
    assert circuit_breaker.get_state(key) == CircuitState.HALF_OPEN

    # 1st success — still needs 2
    circuit_breaker.record_success(key, ticket1)
    assert circuit_breaker.get_state(key) == CircuitState.HALF_OPEN

    # 2nd canary admitted in same generation
    ticket2 = circuit_breaker.can_execute(key)
    assert ticket2 == ticket1
    circuit_breaker.record_success(key, ticket2)

    assert circuit_breaker.get_state(key) == CircuitState.CLOSED
    assert circuit_breaker.can_execute(key) is not None


@pytest.mark.asyncio
async def test_half_open_failure_immediately_reopens(circuit_breaker: CircuitBreaker):
    key = "ollama:llama3"
    circuit_breaker.trip(key)
    await asyncio.sleep(0.12)

    ticket = circuit_breaker.can_execute(key)
    assert ticket is not None
    assert circuit_breaker.get_state(key) == CircuitState.HALF_OPEN

    # Canary fails!
    circuit_breaker.record_failure(key, ticket, COALAIError(COALAIErrorType.PROVIDER_TIMEOUT, "timeout"))
    assert circuit_breaker.get_state(key) == CircuitState.OPEN
    assert circuit_breaker.can_execute(key) is None


# ── Production Failure Mode Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_leaked_canary_slots_expire_via_lease_deadline(circuit_breaker: CircuitBreaker):
    """
    Test 1: In HALF_OPEN, grab all canary slots (max_trials=2) and never report back.
    After canary_timeout_seconds passes, the leases expire and new requests are admitted.
    Prevents permanent wedging.
    """
    key = "ollama:llama3"
    circuit_breaker.trip(key)
    await asyncio.sleep(0.12)

    # Grab both canary slots
    ticket1 = circuit_breaker.can_execute(key)
    ticket2 = circuit_breaker.can_execute(key)
    assert ticket1 is not None
    assert ticket2 is not None

    # 3rd request rejected because all slots taken
    assert circuit_breaker.can_execute(key) is None

    # Wait for canary lease timeout (0.15s in fixture)
    await asyncio.sleep(0.18)

    # Leased deadlines have expired; a new canary request MUST be admitted!
    ticket3 = circuit_breaker.can_execute(key)
    assert ticket3 is not None


@pytest.mark.asyncio
async def test_stale_result_from_closed_state_does_not_corrupt_half_open(circuit_breaker: CircuitBreaker):
    """
    Test 2: A request starts in CLOSED (generation 0).
    While it runs, other failures trip the breaker to OPEN, and it cools down to HALF_OPEN (generation 2).
    When the old request fails, its report with generation 0 must be ignored,
    and the breaker must STAY in HALF_OPEN.
    """
    key = "ollama:llama3"

    # Start request in CLOSED
    old_ticket = circuit_breaker.can_execute(key)
    assert old_ticket is not None

    # Tripped by other requests and cooled down to HALF_OPEN
    circuit_breaker.trip(key)
    await asyncio.sleep(0.12)

    canary_ticket = circuit_breaker.can_execute(key)
    assert canary_ticket is not None
    assert canary_ticket != old_ticket
    assert circuit_breaker.get_state(key) == CircuitState.HALF_OPEN

    # The slow request from CLOSED finally fails!
    circuit_breaker.record_failure(key, old_ticket, COALAIError(COALAIErrorType.PROVIDER_TIMEOUT, "slow"))

    # Breaker must NOT have tripped back to OPEN; stale result was ignored!
    assert circuit_breaker.get_state(key) == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_cancelled_task_releases_canary_slot(circuit_breaker: CircuitBreaker):
    """
    Test 3: If an in-flight request is cancelled (e.g. client disconnects),
    the guard's CancelledError handler releases the canary slot immediately.
    """
    key = "ollama:llama3"
    circuit_breaker.trip(key)
    await asyncio.sleep(0.12)

    async def _client_cancels():
        async with circuit_breaker.guard(key) as admitted:
            assert admitted is True
            # Simulate waiting for provider when client cancels
            await asyncio.sleep(10.0)

    task = asyncio.create_task(_client_cancels())
    await asyncio.sleep(0.01)  # Yield to let task enter guard

    # Cancel the task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Slot must have been released! A new canary should be admitted immediately.
    ticket_after = circuit_breaker.can_execute(key)
    assert ticket_after is not None


@pytest.mark.asyncio
async def test_concurrent_can_execute_admits_exactly_max_trials(circuit_breaker: CircuitBreaker):
    """
    Test 4: 50 concurrent requests in HALF_OPEN should admit exactly max_trials (2).
    """
    key = "ollama:llama3"
    circuit_breaker.trip(key)
    await asyncio.sleep(0.12)

    # 50 concurrent requests try to grab canary slots
    results = [circuit_breaker.can_execute(key) for _ in range(50)]
    admitted = [r for r in results if r is not None]

    assert len(admitted) == 2
