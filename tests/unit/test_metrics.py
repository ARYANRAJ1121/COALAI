"""
Unit tests for the metrics module.

Tests:
  - record_request_complete: counters and histograms increment correctly
  - record_request_complete: never raises even with a broken PipelineContext
  - record_retry: increments retry counter
  - record_provider_error: increments provider error counter
  - record_redis_failure: increments redis failure counter
  - record_rate_limit_exceeded: increments rate limit counter
  - get_metrics_response: returns bytes and correct content-type
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from coalai.models.contracts import (
    ExecutionResult,
    FinishReason,
    Message,
    PipelineContext,
    TenantContext,
    TenantQuotas,
)
from coalai.observability.metrics import (
    CONTENT_TYPE_LATEST,
    accounting_failures_total,
    active_requests,
    get_metrics_response,
    provider_errors_total,
    provider_latency_seconds,
    rate_limit_exceeded_total,
    record_accounting_failure,
    record_provider_error,
    record_rate_limit_exceeded,
    record_redis_failure,
    record_request_complete,
    record_retry,
    redis_failures_total,
    requests_total,
    retry_attempts_total,
    tokens_total,
    ttft_seconds,
)


def _make_pipeline_ctx(
    *,
    stream: bool = False,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> PipelineContext:
    from datetime import timedelta

    tenant_id = uuid.uuid4()
    now = datetime.now(tz=timezone.utc)
    ctx = PipelineContext(
        request_id=uuid.uuid4(),
        trace_id="test-trace",
        received_at=now,
        messages=[Message(role="user", content="Hello")],
        model_hint="llama3.2:3b",
        stream=stream,
        tenant_context=TenantContext(
            tenant_id=tenant_id,
            api_key_id=uuid.uuid4(),
            quotas=TenantQuotas(requests_per_minute=60, requests_per_day=10000),
            enabled_models=[],
        ),
    )
    ctx.result = ExecutionResult(
        success=True,
        content="Hello back",
        finish_reason=FinishReason.STOP,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_used="ollama",
        model_used="llama3.2:3b",
        provider_request_id="chatcmpl-test-123",
        execution_started_at=now,
        execution_completed_at=now + timedelta(milliseconds=500),
        first_token_at=now + timedelta(milliseconds=200) if stream else None,
    )
    ctx.responded_at = now + timedelta(milliseconds=510)
    return ctx


# ── record_request_complete ────────────────────────────────────────────────────


def test_record_request_complete_increments_request_counter() -> None:
    """Successful completion → requests_total increments with status=success."""
    ctx = _make_pipeline_ctx()
    tenant_str = str(ctx.tenant_context.tenant_id)

    before = requests_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", status="success"
    )._value.get()

    record_request_complete(ctx, status="success")

    after = requests_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", status="success"
    )._value.get()

    assert after == before + 1


def test_record_request_complete_increments_token_counters() -> None:
    """Token counters (input + output) increment by the correct amounts."""
    ctx = _make_pipeline_ctx(input_tokens=20, output_tokens=8)
    tenant_str = str(ctx.tenant_context.tenant_id)

    before_in = tokens_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", token_type="input"
    )._value.get()
    before_out = tokens_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", token_type="output"
    )._value.get()

    record_request_complete(ctx, status="success")

    after_in = tokens_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", token_type="input"
    )._value.get()
    after_out = tokens_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", token_type="output"
    )._value.get()

    assert after_in == before_in + 20
    assert after_out == before_out + 8


def test_record_request_complete_records_stream_route_tokens_and_timings() -> None:
    """Stream metrics use the selected fallback target and stream usage fields."""
    from datetime import timedelta

    ctx = _make_pipeline_ctx(stream=True)
    ctx.result = None
    ctx.selected_provider = "groq"
    ctx.selected_model = "llama-3.1-8b-instant"
    ctx.provider_started_at = ctx.received_at + timedelta(milliseconds=50)
    ctx.first_token_at = ctx.received_at + timedelta(milliseconds=250)
    ctx.provider_completed_at = ctx.received_at + timedelta(seconds=1)
    ctx.responded_at = ctx.received_at + timedelta(milliseconds=1100)
    ctx.stream_input_tokens = 20
    ctx.stream_output_tokens = 8

    tenant_id = str(ctx.tenant_id)
    request_before = requests_total.labels(
        tenant_id=tenant_id,
        model=ctx.selected_model,
        provider=ctx.selected_provider,
        status="success",
    )._value.get()
    input_before = tokens_total.labels(
        tenant_id=tenant_id,
        model=ctx.selected_model,
        provider=ctx.selected_provider,
        token_type="input",
    )._value.get()
    output_before = tokens_total.labels(
        tenant_id=tenant_id,
        model=ctx.selected_model,
        provider=ctx.selected_provider,
        token_type="output",
    )._value.get()
    provider_sum_before = provider_latency_seconds.labels(
        provider=ctx.selected_provider, model=ctx.selected_model
    )._sum.get()
    ttft_sum_before = ttft_seconds.labels(
        provider=ctx.selected_provider, model=ctx.selected_model
    )._sum.get()

    record_request_complete(ctx, status="success")

    assert requests_total.labels(
        tenant_id=tenant_id,
        model=ctx.selected_model,
        provider=ctx.selected_provider,
        status="success",
    )._value.get() == request_before + 1
    assert tokens_total.labels(
        tenant_id=tenant_id,
        model=ctx.selected_model,
        provider=ctx.selected_provider,
        token_type="input",
    )._value.get() == input_before + 20
    assert tokens_total.labels(
        tenant_id=tenant_id,
        model=ctx.selected_model,
        provider=ctx.selected_provider,
        token_type="output",
    )._value.get() == output_before + 8
    assert provider_latency_seconds.labels(
        provider=ctx.selected_provider, model=ctx.selected_model
    )._sum.get() == pytest.approx(provider_sum_before + 0.95)
    assert ttft_seconds.labels(
        provider=ctx.selected_provider, model=ctx.selected_model
    )._sum.get() == pytest.approx(ttft_sum_before + 0.2)


def test_record_request_complete_never_raises_on_broken_ctx() -> None:
    """Even if PipelineContext is completely broken, record_* must not raise."""
    broken_ctx = MagicMock()
    broken_ctx.tenant_id = None
    broken_ctx.result = None
    broken_ctx.stream = False
    broken_ctx.model_hint = None
    broken_ctx.total_latency_ms = -1  # invalid
    broken_ctx.tenant_context = None

    # Must not raise
    record_request_complete(broken_ctx, status="error")


def test_record_request_complete_error_status() -> None:
    """Error status is recorded correctly on the counter label."""
    ctx = _make_pipeline_ctx()
    tenant_str = str(ctx.tenant_context.tenant_id)

    before = requests_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", status="provider_timeout"
    )._value.get()

    record_request_complete(ctx, status="provider_timeout")

    after = requests_total.labels(
        tenant_id=tenant_str, model="llama3.2:3b", provider="ollama", status="provider_timeout"
    )._value.get()

    assert after == before + 1


# ── record_retry ───────────────────────────────────────────────────────────────


def test_record_retry_increments_counter() -> None:
    before = retry_attempts_total.labels(
        provider="ollama", error_type="provider_timeout"
    )._value.get()

    record_retry(provider="ollama", error_type="provider_timeout")

    after = retry_attempts_total.labels(
        provider="ollama", error_type="provider_timeout"
    )._value.get()

    assert after == before + 1


# ── record_provider_error ──────────────────────────────────────────────────────


def test_record_provider_error_increments_counter() -> None:
    before = provider_errors_total.labels(
        provider="ollama", error_type="provider_server_error"
    )._value.get()

    record_provider_error(provider="ollama", error_type="provider_server_error")

    after = provider_errors_total.labels(
        provider="ollama", error_type="provider_server_error"
    )._value.get()

    assert after == before + 1


# ── record_redis_failure ───────────────────────────────────────────────────────


def test_record_redis_failure_increments_counter() -> None:
    before = redis_failures_total.labels(
        use_case="auth", operation="get"
    )._value.get()

    record_redis_failure(use_case="auth", operation="get")

    after = redis_failures_total.labels(
        use_case="auth", operation="get"
    )._value.get()

    assert after == before + 1


# ── record_rate_limit_exceeded ────────────────────────────────────────────────


def test_record_rate_limit_exceeded_increments_counter() -> None:
    tenant_id = str(uuid.uuid4())
    before = rate_limit_exceeded_total.labels(tenant_id=tenant_id)._value.get()

    record_rate_limit_exceeded(tenant_id=tenant_id)

    after = rate_limit_exceeded_total.labels(tenant_id=tenant_id)._value.get()

    assert after == before + 1


# ── record_accounting_failure ─────────────────────────────────────────────────


def test_record_accounting_failure_increments_counter() -> None:
    before = accounting_failures_total.labels(phase="confirmed")._value.get()

    record_accounting_failure(phase="confirmed")

    after = accounting_failures_total.labels(phase="confirmed")._value.get()

    assert after == before + 1


# ── get_metrics_response ───────────────────────────────────────────────────────


def test_get_metrics_response_returns_bytes_and_correct_content_type() -> None:
    content, content_type = get_metrics_response()

    assert isinstance(content, bytes)
    assert len(content) > 0
    assert "text/plain" in content_type
    # Must contain at least one of our custom metric names
    assert b"coalai_requests_total" in content
