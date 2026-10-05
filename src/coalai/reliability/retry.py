"""
Retry policy with exponential backoff and jitter.

Design rules enforced here:
  - Retry ONLY on retryable COALAIError types.
  - NEVER retry on: auth failures, bad requests, budget errors, or non-COALAIErrors.
  - ALWAYS apply jitter to prevent retry storms.
  - Respect provider-supplied Retry-After header when present.
  - max_attempts is hard-capped — no infinite loops possible.
  - All retry events are logged and could be metriced.

Usage:
    result = await retry_with_backoff(
        fn=lambda: provider.complete(request, timeout_seconds=30.0),
        policy=RetryPolicy(),
    )
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from coalai.config import settings
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration. Defaults are loaded from application settings."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    # Random jitter applied as ± fraction of the computed delay.
    # 0.25 = ±25%. Mandatory — retries without jitter cause retry storms.
    jitter_factor: float = 0.25
    # Maximum delay cap regardless of backoff computation.
    max_delay_seconds: float = 30.0

    @classmethod
    def from_settings(cls) -> "RetryPolicy":
        return cls(
            max_attempts=settings.retry_max_attempts,
            initial_delay_seconds=settings.retry_initial_delay_seconds,
            backoff_factor=settings.retry_backoff_factor,
            jitter_factor=settings.retry_jitter_factor,
        )

    def compute_delay(self, attempt: int) -> float:
        """
        Compute the delay before attempt N+1.
        attempt=1 means the first retry (after the first failure).

        Delay = initial_delay * backoff_factor^(attempt-1)
                ± jitter_factor * delay (random, uniform)
        """
        base_delay = self.initial_delay_seconds * (self.backoff_factor ** (attempt - 1))
        base_delay = min(base_delay, self.max_delay_seconds)
        jitter_range = base_delay * self.jitter_factor
        delay = base_delay + random.uniform(-jitter_range, jitter_range)
        return max(0.0, delay)


def _is_retryable(exc: COALAIError) -> bool:
    """
    Determine if a COALAIError warrants a retry.

    Non-retryable errors are those where retrying will produce the same
    result (e.g., auth failures, bad requests). Retrying them wastes time
    and amplifies provider load.
    """
    return exc.retryable


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    request_id: str = "",
) -> T:
    """
    Execute fn() with retry logic.

    - Retries only on retryable COALAIError.
    - Does NOT retry on non-COALAIError exceptions (programming errors, etc.).
    - Returns the result of the first successful call.
    - Raises the last COALAIError after all attempts are exhausted.

    Args:
        fn: Async callable to execute (no arguments; use a lambda or partial).
        policy: RetryPolicy to use. Defaults to settings-based policy.
        request_id: For logging correlation.
    """
    if policy is None:
        policy = RetryPolicy.from_settings()

    last_error: COALAIError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()

        except COALAIError as exc:
            last_error = exc

            if not _is_retryable(exc):
                log.debug(
                    "retry_skipped_non_retryable",
                    request_id=request_id,
                    attempt=attempt,
                    error_type=exc.error_type,
                )
                raise

            if attempt == policy.max_attempts:
                log.warning(
                    "retry_exhausted",
                    request_id=request_id,
                    max_attempts=policy.max_attempts,
                    error_type=exc.error_type,
                )
                raise

            # Honour provider-specified Retry-After if present (capped at max_delay).
            if exc.retry_after_seconds is not None:
                delay = min(exc.retry_after_seconds, policy.max_delay_seconds)
                log.info(
                    "retry_honouring_retry_after",
                    request_id=request_id,
                    attempt=attempt,
                    delay_seconds=round(delay, 2),
                    error_type=exc.error_type,
                )
            else:
                delay = policy.compute_delay(attempt)
                log.info(
                    "retry_scheduled",
                    request_id=request_id,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                    error_type=exc.error_type,
                )

            await asyncio.sleep(delay)

        except Exception:
            # Non-COALAIError: programming bug or unexpected failure.
            # Do NOT retry — re-raise immediately.
            raise

    # This line is unreachable — the loop always returns or raises.
    # Added to satisfy type checkers.
    assert last_error is not None
    raise last_error
