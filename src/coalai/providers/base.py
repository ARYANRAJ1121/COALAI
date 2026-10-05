"""
LLMProvider protocol and associated types.

All provider adapters implement this protocol. The Reliability Engine
operates exclusively against this protocol — it never calls provider-specific
code directly.

The protocol defines three methods:
  - complete()  → non-streaming call → ExecutionResult
  - stream()    → streaming call → AsyncIterator[NormalizedChunk]
  - health_check() → ProviderHealth

Error contract:
  Provider adapters MUST catch their own exceptions and raise COALAIError
  before returning to the caller. No provider-specific exception type
  (httpx.TimeoutException, aiohttp.ClientError, etc.) may cross the
  adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import AsyncIterator, Protocol, runtime_checkable

from coalai.models.contracts import ExecutionResult, NormalizedChunk, NormalizedRequest


class ProviderStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class ProviderHealth:
    provider_id: str
    status: ProviderStatus
    latency_ms: float | None  # None if health check timed out
    error: str | None = None  # Human-readable error, if status != HEALTHY


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol that all LLM provider adapters must satisfy.

    provider_id must be a stable, unique string identifier
    (e.g., "ollama", "groq", "openai"). It is used in:
      - Routing decisions
      - Circuit breaker state keys
      - Accounting ledger records
      - Observability span attributes
    """

    provider_id: str

    async def complete(
        self,
        request: NormalizedRequest,
        *,
        timeout_seconds: float,
    ) -> ExecutionResult:
        """
        Execute a non-streaming chat completion.

        Must return an ExecutionResult with success=True on success.
        Must raise COALAIError (never a provider-specific exception) on failure.
        Must record attempt timing precisely (execution_started_at → execution_completed_at).
        """
        ...

    async def stream(
        self,
        request: NormalizedRequest,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[NormalizedChunk]:
        """
        Execute a streaming chat completion.

        Yields NormalizedChunk objects until the stream is exhausted.
        The final chunk must set finish_reason.
        When token counts are available (from the final chunk or accumulated),
        they must be set on the last chunk.
        Must raise COALAIError on failure before yielding any chunk.
        """
        ...

    async def health_check(self) -> ProviderHealth:
        """
        Probe the provider's availability.
        Must return a ProviderHealth within provider_health_check_timeout_seconds.
        Must not raise — catch all exceptions and return ProviderHealth(status=UNREACHABLE).
        """
        ...
