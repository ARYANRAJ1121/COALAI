"""
Prometheus metrics for COALAI — Milestone 2.

All metrics are defined here as module-level singletons.
No other module imports prometheus_client directly.

Usage pattern:
    from coalai.observability.metrics import record_request_complete
    record_request_complete(ctx, status="success")

Design rules:
  - Labels must be low-cardinality (no request_id, no raw content).
  - tenant_id is included for per-tenant dashboards but must be
    verified (from TenantContext) — never from raw request data.
  - All record_* functions are synchronous and non-blocking.
    They never raise — a metrics failure must never affect a request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from coalai.models.contracts import PipelineContext

# ── Latency buckets (seconds) ─────────────────────────────────────────────────
# Tuned for LLM workloads: from sub-100ms (cache hit) to 120s (long generation)
_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0
)

_TTFT_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0
)

_TOKEN_BUCKETS = (
    10, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000
)

# ── Request metrics ───────────────────────────────────────────────────────────

requests_total = Counter(
    "coalai_requests_total",
    "Total number of COALAI requests by outcome.",
    ["tenant_id", "model", "provider", "status"],   # status: success | error_type
)

request_duration_seconds = Histogram(
    "coalai_request_duration_seconds",
    "End-to-end gateway latency (client in → response out).",
    ["model", "provider", "stream"],
    buckets=_LATENCY_BUCKETS,
)

provider_latency_seconds = Histogram(
    "coalai_provider_latency_seconds",
    "Time spent waiting for the provider (excludes gateway overhead).",
    ["provider", "model"],
    buckets=_LATENCY_BUCKETS,
)

# ── Token metrics ─────────────────────────────────────────────────────────────

tokens_total = Counter(
    "coalai_tokens_total",
    "Total tokens processed (input and output separately).",
    ["tenant_id", "model", "provider", "token_type"],   # token_type: input | output
)

tokens_per_request = Histogram(
    "coalai_tokens_per_request",
    "Token count distribution per request.",
    ["model", "token_type"],
    buckets=_TOKEN_BUCKETS,
)

# ── Streaming metrics ─────────────────────────────────────────────────────────

ttft_seconds = Histogram(
    "coalai_ttft_seconds",
    "Time-to-first-token for streaming requests.",
    ["provider", "model"],
    buckets=_TTFT_BUCKETS,
)

# ── Reliability metrics ───────────────────────────────────────────────────────

retry_attempts_total = Counter(
    "coalai_retry_attempts_total",
    "Number of provider retry attempts.",
    ["provider", "error_type"],
)

provider_errors_total = Counter(
    "coalai_provider_errors_total",
    "Provider errors by type (after all retries exhausted).",
    ["provider", "error_type"],
)

# ── Cache metrics (prep for M3) ───────────────────────────────────────────────

cache_hits_total = Counter(
    "coalai_cache_hits_total",
    "Cache hits by cache type.",
    ["cache_type"],   # exact | semantic
)

cache_misses_total = Counter(
    "coalai_cache_misses_total",
    "Cache misses by cache type.",
    ["cache_type"],
)

# ── Infrastructure health metrics ─────────────────────────────────────────────

redis_failures_total = Counter(
    "coalai_redis_failures_total",
    "Redis operation failures by use case (fail-open incidents).",
    ["use_case", "operation"],   # operation: get | set | incr | delete
)

accounting_failures_total = Counter(
    "coalai_accounting_failures_total",
    "Accounting write failures (triggers 503 for confirmed, logged for pending confirm).",
    ["phase"],   # confirmed | pending | confirm
)

auth_cache_hits_total = Counter(
    "coalai_auth_cache_hits_total",
    "Auth key cache hits (Redis fast path).",
)

auth_cache_misses_total = Counter(
    "coalai_auth_cache_misses_total",
    "Auth key cache misses (fell back to Postgres).",
)

rate_limit_exceeded_total = Counter(
    "coalai_rate_limit_exceeded_total",
    "Rate limit rejections.",
    ["tenant_id"],
)

# ── Active requests gauge ─────────────────────────────────────────────────────

active_requests = Gauge(
    "coalai_active_requests",
    "Number of requests currently in-flight.",
)


# ── High-level record helpers ─────────────────────────────────────────────────


def record_request_complete(ctx: "PipelineContext", *, status: str) -> None:
    """
    Record all metrics for a completed request.

    Call this once per request after the response has been sent.
    status = "success" | error_type string (e.g. "provider_timeout")
    Never raises — metric failures must not affect request handling.
    """
    try:
        tenant_id = str(ctx.tenant_id) if ctx.tenant_id else "unknown"
        provider = ctx.result.provider_used if ctx.result else (ctx.selected_provider or "unknown")
        model = ctx.result.model_used if ctx.result else (ctx.selected_model or ctx.model_hint or "unknown")
        stream = str(ctx.stream).lower()

        # Request counter
        requests_total.labels(
            tenant_id=tenant_id,
            model=model,
            provider=provider,
            status=status,
        ).inc()

        # Latency histograms
        total_latency = ctx.total_latency_ms / 1000.0
        if total_latency > 0:
            request_duration_seconds.labels(
                model=model, provider=provider, stream=stream
            ).observe(total_latency)

        if ctx.result or ctx.provider_started_at:
            prov_latency = ctx.provider_latency_ms / 1000.0
            if prov_latency > 0:
                provider_latency_seconds.labels(
                    provider=provider, model=model
                ).observe(prov_latency)

        # Token counters (buffered and streaming requests).
        if ctx.input_tokens:
            tokens_total.labels(
                tenant_id=tenant_id, model=model, provider=provider, token_type="input"
            ).inc(ctx.input_tokens)
            tokens_per_request.labels(model=model, token_type="input").observe(ctx.input_tokens)

        if ctx.output_tokens:
            tokens_total.labels(
                tenant_id=tenant_id, model=model, provider=provider, token_type="output"
            ).inc(ctx.output_tokens)
            tokens_per_request.labels(model=model, token_type="output").observe(ctx.output_tokens)

        # TTFT (streaming only).
        if ctx.stream and ctx.first_token_at and ctx.provider_started_at:
            ttft_ms = (ctx.first_token_at - ctx.provider_started_at).total_seconds() * 1000.0
            ttft_seconds.labels(provider=provider, model=model).observe(ttft_ms / 1000.0)

    except Exception:
        # Metrics must never crash a request
        pass


def record_retry(*, provider: str, error_type: str) -> None:
    """Record a single retry attempt."""
    try:
        retry_attempts_total.labels(provider=provider, error_type=error_type).inc()
    except Exception:
        pass


def record_provider_error(*, provider: str, error_type: str) -> None:
    """Record a terminal provider error (after all retries)."""
    try:
        provider_errors_total.labels(provider=provider, error_type=error_type).inc()
    except Exception:
        pass


def record_redis_failure(*, use_case: str, operation: str) -> None:
    """Record a Redis fail-open incident."""
    try:
        redis_failures_total.labels(use_case=use_case, operation=operation).inc()
    except Exception:
        pass


def record_accounting_failure(*, phase: str) -> None:
    """Record an accounting write failure."""
    try:
        accounting_failures_total.labels(phase=phase).inc()
    except Exception:
        pass


def record_rate_limit_exceeded(*, tenant_id: str) -> None:
    """Record a rate limit rejection."""
    try:
        rate_limit_exceeded_total.labels(tenant_id=tenant_id).inc()
    except Exception:
        pass


def get_metrics_response() -> tuple[bytes, str]:
    """Return (content, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
