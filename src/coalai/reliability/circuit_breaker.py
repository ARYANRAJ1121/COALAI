"""
Circuit breaker for upstream providers and models.

States:
  - CLOSED: Normal operation. Requests flow freely.
  - OPEN: Tripped due to consecutive failures. Requests fail fast (or route to fallback).
  - HALF_OPEN: Recovery probe. Limited test requests (canaries) are sent to check if upstream recovered.

Design Rules:
  - Event-loop atomic: Critical sections are purely synchronous and free of await points,
    preventing race conditions on the single-threaded asyncio event loop without locking overhead.
  - Generation Tickets: Every state transition increments `generation`. Reports (success/failure/release)
    require a valid ticket; stale results from previous states are safely ignored.
  - Leased Canary Deadlines: Canary slots have an expiry deadline. If a request hangs, is cancelled,
    or drops without reporting, the slot automatically expires, eliminating permanent wedging.
  - Context Manager Guard: `async with breaker.guard(key) as permitted:` encapsulates admission,
    reporting, and cancellation release in a single bulletproof block.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import AsyncIterator

from coalai.config import settings
from coalai.models.errors import COALAIError
from coalai.observability.logging import get_logger

log = get_logger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_time_seconds: float = 30.0
    half_open_success_threshold: int = 2
    half_open_max_trials: int = 2
    canary_timeout_seconds: float = 10.0

    @classmethod
    def from_settings(cls) -> CircuitBreakerConfig:
        return cls(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_time_seconds=settings.circuit_breaker_recovery_time_seconds,
            half_open_success_threshold=settings.circuit_breaker_half_open_success_threshold,
            half_open_max_trials=settings.circuit_breaker_half_open_max_trials,
            canary_timeout_seconds=settings.circuit_breaker_canary_timeout_seconds,
        )


@dataclass
class _CircuitEntry:
    state: CircuitState = CircuitState.CLOSED
    generation: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None
    trial_deadlines: list[float] = field(default_factory=list)


class CircuitBreaker:
    """Asyncio-optimized circuit breaker with generation tickets and leased canary probes."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig.from_settings()
        self._entries: dict[str, _CircuitEntry] = {}

    def _get_entry(self, key: str) -> _CircuitEntry:
        if key not in self._entries:
            self._entries[key] = _CircuitEntry()
        return self._entries[key]

    def get_state(self, key: str) -> CircuitState:
        """
        Pure read-only query of circuit state without mutating entries.
        Reports HALF_OPEN if OPEN and recovery time has elapsed.
        """
        if key not in self._entries:
            return CircuitState.CLOSED

        entry = self._entries[key]
        if entry.state == CircuitState.OPEN and entry.opened_at is not None:
            if (time.monotonic() - entry.opened_at) >= self.config.recovery_time_seconds:
                return CircuitState.HALF_OPEN
        return entry.state

    def can_execute(self, key: str) -> int | None:
        """
        Check if execution is admitted.
        Returns generation ticket (int) if admitted, or None if rejected.
        """
        entry = self._get_entry(key)
        now = time.monotonic()

        if entry.state == CircuitState.CLOSED:
            return entry.generation

        if entry.state == CircuitState.OPEN:
            if entry.opened_at is not None and (now - entry.opened_at) >= self.config.recovery_time_seconds:
                # Transition to HALF_OPEN
                entry.state = CircuitState.HALF_OPEN
                entry.generation += 1
                entry.consecutive_successes = 0
                entry.trial_deadlines = [now + self.config.canary_timeout_seconds]
                log.info(
                    "circuit_state_transition",
                    key=key,
                    old_state=CircuitState.OPEN,
                    new_state=CircuitState.HALF_OPEN,
                    generation=entry.generation,
                    reason="recovery_time_elapsed",
                )
                return entry.generation
            return None

        if entry.state == CircuitState.HALF_OPEN:
            # Expire leaked/abandoned canary leases
            entry.trial_deadlines = [d for d in entry.trial_deadlines if d > now]
            if len(entry.trial_deadlines) < self.config.half_open_max_trials:
                entry.trial_deadlines.append(now + self.config.canary_timeout_seconds)
                return entry.generation
            return None

        return None

    def record_success(self, key: str, ticket: int) -> None:
        """Record success for a specific generation ticket."""
        if key not in self._entries:
            return

        entry = self._entries[key]
        if ticket != entry.generation:
            # Stale response from a previous state generation — ignore
            return

        if entry.state == CircuitState.HALF_OPEN:
            if entry.trial_deadlines:
                entry.trial_deadlines.pop(0)
            entry.consecutive_successes += 1
            if entry.consecutive_successes >= self.config.half_open_success_threshold:
                entry.state = CircuitState.CLOSED
                entry.generation += 1
                entry.consecutive_failures = 0
                entry.consecutive_successes = 0
                entry.opened_at = None
                entry.trial_deadlines.clear()
                log.info(
                    "circuit_state_transition",
                    key=key,
                    old_state=CircuitState.HALF_OPEN,
                    new_state=CircuitState.CLOSED,
                    generation=entry.generation,
                    reason="canary_success_threshold_reached",
                )
        elif entry.state == CircuitState.CLOSED:
            entry.consecutive_failures = 0

    def record_failure(self, key: str, ticket: int, exc: Exception | None = None) -> None:
        """
        Record failure for a specific generation ticket.
        Client errors (non-retryable COALAIError) free the slot without tripping the breaker.
        """
        if isinstance(exc, COALAIError) and not exc.retryable:
            self.release(key, ticket)
            return

        if key not in self._entries:
            return

        entry = self._entries[key]
        if ticket != entry.generation:
            # Stale failure from a previous state generation — ignore
            return

        now = time.monotonic()

        if entry.state == CircuitState.HALF_OPEN:
            entry.state = CircuitState.OPEN
            entry.generation += 1
            entry.opened_at = now
            entry.trial_deadlines.clear()
            log.warning(
                "circuit_state_transition",
                key=key,
                old_state=CircuitState.HALF_OPEN,
                new_state=CircuitState.OPEN,
                generation=entry.generation,
                reason="canary_trial_failed",
            )
        elif entry.state == CircuitState.CLOSED:
            entry.consecutive_failures += 1
            if entry.consecutive_failures >= self.config.failure_threshold:
                entry.state = CircuitState.OPEN
                entry.generation += 1
                entry.opened_at = now
                entry.trial_deadlines.clear()
                log.warning(
                    "circuit_state_transition",
                    key=key,
                    old_state=CircuitState.CLOSED,
                    new_state=CircuitState.OPEN,
                    generation=entry.generation,
                    consecutive_failures=entry.consecutive_failures,
                    threshold=self.config.failure_threshold,
                )

    def release(self, key: str, ticket: int) -> None:
        """Release a canary slot without counting success or failure (e.g. cancellation)."""
        if key not in self._entries:
            return
        entry = self._entries[key]
        if ticket != entry.generation:
            return
        if entry.state == CircuitState.HALF_OPEN:
            if entry.trial_deadlines:
                entry.trial_deadlines.pop(0)

    def trip(self, key: str) -> None:
        """Manually trip circuit breaker to OPEN."""
        entry = self._get_entry(key)
        entry.state = CircuitState.OPEN
        entry.generation += 1
        entry.opened_at = time.monotonic()
        entry.trial_deadlines.clear()

    def reset(self, key: str) -> None:
        """Reset circuit breaker to CLOSED."""
        entry = self._get_entry(key)
        entry.state = CircuitState.CLOSED
        entry.generation += 1
        entry.consecutive_failures = 0
        entry.consecutive_successes = 0
        entry.opened_at = None
        entry.trial_deadlines.clear()

    @asynccontextmanager
    async def guard(self, key: str) -> AsyncIterator[bool]:
        """
        Async context manager guarding execution against the circuit breaker.
        Yields True if admitted, False if rejected (OPEN or max trials reached).
        Handles success, failure, and cancellation release automatically.
        """
        ticket = self.can_execute(key)
        if ticket is None:
            yield False
            return

        try:
            yield True
        except asyncio.CancelledError:
            self.release(key, ticket)
            raise
        except COALAIError as exc:
            if not exc.retryable:
                self.release(key, ticket)
            else:
                self.record_failure(key, ticket, exc)
            raise
        except Exception as exc:
            self.release(key, ticket)
            raise
        else:
            self.record_success(key, ticket)
