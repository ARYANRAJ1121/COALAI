"""
Request and response Pydantic schemas for the COALAI gateway.

These are the HTTP-layer types — they are distinct from the domain types
in coalai.models.contracts. The gateway translates between these schemas
and the domain contracts; no other module uses these schemas.

Validation rules enforced here:
  - temperature: [0.0, 2.0]
  - max_tokens: [1, 128000]
  - messages: at least 1 message, valid roles, non-empty content
  - model: string, no injection surface (plain string)

What is NOT done here:
  - Budget enforcement (Phase 7)
  - Prompt injection detection (not in MVP — not claimed anywhere)
  - Model existence check (done by the router against the registry)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request types ─────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    COALAI accepts a superset of the standard fields.
    """

    model: str | None = Field(
        default=None,
        description="Model to use. If omitted, the gateway uses the configured default.",
        examples=["llama3.2:3b"],
    )
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=128_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    seed: int | None = None
    # COALAI extension: client-provided idempotency key for safe retries
    idempotency_key: str | None = Field(default=None, alias="coalai_idempotency_key")

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user_or_tool(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """The conversation must end with a user or tool message."""
        if v and v[-1].role not in ("user", "tool"):
            raise ValueError(
                "The last message must have role 'user' or 'tool'. "
                f"Got: '{v[-1].role}'"
            )
        return v

    @field_validator("stop")
    @classmethod
    def normalise_stop(cls, v: list[str] | str | None) -> list[str] | None:
        """Normalise stop to a list or None."""
        if isinstance(v, str):
            return [v]
        return v

    model_config = {"populate_by_name": True}


# ── Response types ─────────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class COALAIResponseExtension(BaseModel):
    """
    COALAI-specific metadata included in every response.
    Clients can use this for observability, debugging, and cost tracking.
    """

    request_id: str
    trace_id: str
    provider_used: str
    model_used: str
    fallback_triggered: bool
    attempt_number: int
    cache_hit: bool
    cache_type: Literal["exact", "semantic"] | None
    estimated_cost_usd: float
    gateway_latency_ms: float
    provider_latency_ms: float
    total_latency_ms: float
    ttft_ms: float | None  # Time-to-first-token; None for non-streaming


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response with a COALAI extension field."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    coalai: COALAIResponseExtension


# ── Streaming chunk types ──────────────────────────────────────────────────────


class StreamingDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: StreamingDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    model: str
    choices: list[StreamingChoice]


# ── Health response types ──────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    postgres: bool
    redis: bool


# ── Error response type ────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    error_type: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
