"""
Core domain contracts.

These types define the data that flows between pipeline stages.
They are the single source of truth for what a "request" and a "result" mean
inside COALAI — independent of HTTP, database, or provider representations.

Design rules:
  - Types used as pipeline inputs/outputs are frozen (immutable).
  - ExecutionTrace is mutable — it is a running record assembled across stages.
  - No FastAPI, SQLAlchemy, or httpx types appear here.
  - No provider-specific types appear here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

if TYPE_CHECKING:
    from coalai.models.errors import COALAIErrorType


# ── Enumerations ──────────────────────────────────────────────────────────────


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    CANCELLED = "cancelled"


# ── Primitive message types ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Message:
    """A single turn in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


# ── Tenant context (populated by Auth module after key verification) ───────────


@dataclass(frozen=True)
class TenantQuotas:
    """
    Rate and budget limits for a tenant.

    NOTE (Phase constraint): monetary budget fields are stored here
    so the data model is complete, but they are NOT enforced until Phase 7.
    Only requests_per_minute is enforced in Milestone 1 (via Redis counter).
    """

    requests_per_minute: int
    requests_per_day: int
    # Phase 7+: monetary enforcement. Present in schema but ignored until then.
    monthly_budget_usd: float | None = None


@dataclass(frozen=True)
class TenantContext:
    """
    Verified identity and limits for the tenant making this request.
    Only populated AFTER the auth module successfully verifies the API key.
    NEVER derived from untrusted request data.
    """

    tenant_id: UUID
    api_key_id: UUID
    quotas: TenantQuotas
    # Allowlist of model IDs this tenant may request.
    # Empty list = all models permitted.
    enabled_models: list[str]


# ── Provider-layer types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class NormalizedRequest:
    """
    Provider-agnostic request. Created by the pipeline from the client request
    and passed into LLMProvider adapters. Contains no HTTP or database types.
    """

    messages: list[Message]
    model: str  # Provider-specific model name, already resolved
    stream: bool
    request_id: UUID
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None
    seed: int | None = None


@dataclass(frozen=True)
class NormalizedChunk:
    """A single chunk from a streaming provider response."""

    content: str
    finish_reason: FinishReason | None
    # Populated only in the final chunk (provider-dependent).
    input_tokens: int | None = None
    output_tokens: int | None = None


# ── Execution result (output of the Reliability Engine) ───────────────────────


@dataclass(frozen=True)
class ExecutionResult:
    """
    Normalized output of one provider call attempt (success or failure).
    Produced by the Reliability Engine after all retry/fallback logic is applied.

    One ExecutionResult is created per attempt. The final result (the one
    returned to the client) is stored in ExecutionTrace.result; all attempts
    (including failed retries) are stored in ExecutionTrace.all_attempts.
    """

    success: bool
    content: str | None  # None when success=False
    finish_reason: FinishReason

    # Token accounting — source of truth for cost calculation.
    # Kept separate because providers may charge different rates for each.
    input_tokens: int
    output_tokens: int

    # Provider identity — which provider/model actually handled this attempt.
    provider_used: str
    model_used: str
    # Provider's own request ID, for support correlation. May be None if the
    # provider does not return one (e.g., connection timeout before response).
    provider_request_id: str | None

    # Timing
    execution_started_at: datetime
    execution_completed_at: datetime
    first_token_at: datetime | None = None  # Streaming only; used to compute TTFT

    # Reliability metadata
    attempt_number: int = 1  # 1 = first attempt, 2 = first retry, etc.

    # Error info (populated when success=False)
    error_type: COALAIErrorType | None = None
    error_message: str | None = None
    retryable: bool = False

    # Raw provider response, kept for debugging. Never forwarded to clients.
    raw_provider_response: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def provider_latency_ms(self) -> float:
        delta = self.execution_completed_at - self.execution_started_at
        return delta.total_seconds() * 1000.0

    @property
    def ttft_ms(self) -> float | None:
        """Time-to-first-token in milliseconds. None for non-streaming calls."""
        if self.first_token_at is None:
            return None
        delta = self.first_token_at - self.execution_started_at
        return delta.total_seconds() * 1000.0


# ── Pipeline context (mutable, assembled across pipeline stages) ───────────────


@dataclass
class PipelineContext:
    """
    Mutable context that accumulates state as a request flows through the
    COALAI pipeline. One instance is created per request.

    Stages read from this context and write their outputs back into it.
    The context is never passed across network/process boundaries — it is
    entirely in-process state for one request's lifetime.
    """

    # ── Identity (set at gateway entry, before any pipeline stage) ────────────
    request_id: UUID
    trace_id: str
    received_at: datetime

    # ── Raw request data (from client, before auth) ───────────────────────────
    messages: list[Message]
    model_hint: str | None  # What the client asked for; a hint, not a mandate
    stream: bool
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    idempotency_key: str | None = None

    # ── Set by Auth module (post-verification only) ───────────────────────────
    tenant_context: TenantContext | None = None

    # ── Set by provider execution ─────────────────────────────────────────────
    result: ExecutionResult | None = None
    all_attempts: list[ExecutionResult] = field(default_factory=list)

    # ── Set by accounting module ──────────────────────────────────────────────
    # v0.3 Option A: accounting_status is None only if the request returned 503
    # before any accounting write was attempted.
    accounting_event_id: UUID | None = None
    accounting_status: Literal["CONFIRMED", "PENDING"] | None = None

    # ── Outcome (set by gateway after pipeline completes) ─────────────────────
    http_status: int = 0
    responded_at: datetime | None = None

    @property
    def tenant_id(self) -> UUID | None:
        """
        Convenience accessor. Returns None before auth completes.
        Use this in log entries ONLY after auth has populated tenant_context.
        """
        return self.tenant_context.tenant_id if self.tenant_context else None

    @property
    def total_latency_ms(self) -> float:
        if self.responded_at is None:
            return 0.0
        return (self.responded_at - self.received_at).total_seconds() * 1000.0

    @property
    def provider_latency_ms(self) -> float:
        return self.result.provider_latency_ms if self.result else 0.0

    @property
    def input_tokens(self) -> int:
        return self.result.input_tokens if self.result else 0

    @property
    def output_tokens(self) -> int:
        return self.result.output_tokens if self.result else 0

    @property
    def estimated_cost_usd(self) -> float:
        """
        Always 0.0 until Phase 7 (budget enforcement) when real pricing is
        looked up from the model registry. Recorded in the accounting ledger
        for observability even when enforcement is disabled.
        """
        return 0.0


def utcnow() -> datetime:
    """Return current UTC time with timezone info. Use instead of datetime.utcnow()."""
    return datetime.now(tz=timezone.utc)
