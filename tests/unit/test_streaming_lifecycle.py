"""Regression coverage for streaming completion, accounting, and request metrics."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from coalai.gateway import router as gateway_router
from coalai.gateway.schemas import ChatCompletionRequest
from coalai.models.contracts import (
    FinishReason,
    NormalizedChunk,
    TenantContext,
    TenantQuotas,
    utcnow,
)
from coalai.observability.metrics import active_requests, requests_total
from coalai.providers.registry import RouteTarget


class StubRegistry:
    def resolve_targets(self, _: str | None) -> list[RouteTarget]:
        return [RouteTarget(provider_id="ollama", model="llama3.2:3b")]


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer test-key"), (b"host", b"test")],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
            "app": FastAPI(),
        }
    )


@pytest.mark.asyncio
async def test_stream_finalizes_after_body_and_uses_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = uuid.uuid4()
    tenant = TenantContext(
        tenant_id=tenant_id,
        api_key_id=uuid.uuid4(),
        quotas=TenantQuotas(requests_per_minute=60, requests_per_day=1000),
        enabled_models=[],
    )
    selected_model = "llama-3.1-8b-instant"
    context_box: dict[str, object] = {}
    event_id = uuid.uuid4()
    session = MagicMock()
    session.commit = AsyncMock()

    async def authenticate(ctx, _raw_key, _session) -> None:
        ctx.tenant_context = tenant
        context_box["ctx"] = ctx

    async def stream(ctx, **_) -> AsyncIterator[NormalizedChunk]:
        ctx.selected_provider = "groq"
        ctx.selected_model = selected_model
        ctx.provider_started_at = utcnow()

        async def chunks() -> AsyncIterator[NormalizedChunk]:
            yield NormalizedChunk(content="answer", finish_reason=None)
            yield NormalizedChunk(
                content="",
                finish_reason=FinishReason.STOP,
                input_tokens=4,
                output_tokens=2,
            )

        return chunks()

    async def write_pending(ctx, _estimated_input_tokens, _session):
        assert ctx.selected_provider == "groq"
        assert ctx.selected_model == selected_model
        return event_id

    @asynccontextmanager
    async def confirm_session():
        yield MagicMock()

    monkeypatch.setattr(gateway_router.pipeline, "run_auth", authenticate)
    monkeypatch.setattr(gateway_router.pipeline, "run_rate_limit", AsyncMock())
    monkeypatch.setattr(gateway_router.pipeline, "execute_stream", stream)
    monkeypatch.setattr(gateway_router.pipeline, "run_accounting_pending", write_pending)
    monkeypatch.setattr(gateway_router, "_get_provider_registry", lambda _request: StubRegistry())
    monkeypatch.setattr(gateway_router, "_get_fallback_router", lambda _request: object())
    monkeypatch.setattr(gateway_router, "_get_provider", lambda _request: object())
    monkeypatch.setattr(gateway_router, "_get_retry_policy", lambda _request: object())
    monkeypatch.setattr(gateway_router, "get_session", confirm_session)
    confirm = AsyncMock()
    monkeypatch.setattr(gateway_router, "confirm_pending", confirm)

    request = _request()
    request_model = ChatCompletionRequest(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    before_active = active_requests._value.get()
    before_requests = requests_total.labels(
        tenant_id=str(tenant_id),
        model=selected_model,
        provider="groq",
        status="success",
    )._value.get()

    response = await gateway_router.chat_completions(request_model, request, session)

    assert active_requests._value.get() == before_active + 1
    ctx = context_box["ctx"]
    assert ctx.responded_at is None

    body_parts = [part async for part in response.body_iterator]
    body = "".join(part.decode() if isinstance(part, bytes) else part for part in body_parts)

    assert "answer" in body
    assert selected_model in body
    assert "data: [DONE]" in body
    assert ctx.responded_at is not None
    assert active_requests._value.get() == before_active
    assert requests_total.labels(
        tenant_id=str(tenant_id),
        model=selected_model,
        provider="groq",
        status="success",
    )._value.get() == before_requests + 1
    confirm.assert_awaited_once()
    assert confirm.await_args.kwargs["actual_input_tokens"] == 4
    assert confirm.await_args.kwargs["actual_output_tokens"] == 2
