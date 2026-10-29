"""
Unit tests for the Groq provider adapter.

All tests use respx to mock httpx — no real Groq API calls are made.

Tests:
  - Successful non-streaming completion
  - HTTP 429 → PROVIDER_RATE_LIMITED with retry_after_seconds
  - HTTP 500 → PROVIDER_SERVER_ERROR (retryable)
  - HTTP 400 → PROVIDER_BAD_REQUEST (not retryable)
  - HTTP 401 → PROVIDER_AUTH_FAILURE (not retryable)
  - Connection timeout → PROVIDER_TIMEOUT (retryable)
  - Malformed JSON response → PROVIDER_MALFORMED_RESPONSE (not retryable)
  - Streaming: chunks yielded correctly
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import respx

from coalai.models.contracts import FinishReason, Message, NormalizedRequest
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.providers.groq import GroqProvider

BASE_URL = "https://test-groq.example.com"


def make_provider() -> GroqProvider:
    return GroqProvider(
        api_key="test-groq-key",
        base_url=BASE_URL,
        default_model="llama-3.1-8b-instant",
    )


def make_request(stream: bool = False) -> NormalizedRequest:
    return NormalizedRequest(
        messages=[Message(role="user", content="Say hello")],
        model="llama-3.1-8b-instant",
        stream=stream,
        request_id=uuid.uuid4(),
        max_tokens=50,
        temperature=0.0,
    )


SUCCESSFUL_RESPONSE = {
    "id": "chatcmpl-groq-test-123",
    "object": "chat.completion",
    "model": "llama-3.1-8b-instant",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
}


# ── Successful non-streaming completion ───────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_success() -> None:
    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=SUCCESSFUL_RESPONSE)
    )
    provider = make_provider()
    result = await provider.complete(make_request(), timeout_seconds=10.0)
    await provider.close()

    assert result.success is True
    assert result.content == "Hello!"
    assert result.finish_reason == FinishReason.STOP
    assert result.input_tokens == 10
    assert result.output_tokens == 3
    assert result.provider_used == "groq"
    assert result.model_used == "llama-3.1-8b-instant"
    assert result.provider_request_id == "chatcmpl-groq-test-123"


# ── HTTP 429 → PROVIDER_RATE_LIMITED ──────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_429_rate_limited() -> None:
    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded"}},
            headers={"retry-after": "10"},
        )
    )
    provider = make_provider()
    with pytest.raises(COALAIError) as exc_info:
        await provider.complete(make_request(), timeout_seconds=10.0)
    await provider.close()

    err = exc_info.value
    assert err.error_type == COALAIErrorType.PROVIDER_RATE_LIMITED
    assert err.retryable is True
    assert err.retry_after_seconds == 10.0


# ── HTTP 500 → PROVIDER_SERVER_ERROR ─────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_500_server_error() -> None:
    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "internal error"}})
    )
    provider = make_provider()
    with pytest.raises(COALAIError) as exc_info:
        await provider.complete(make_request(), timeout_seconds=10.0)
    await provider.close()

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_SERVER_ERROR
    assert exc_info.value.retryable is True


# ── HTTP 400 → PROVIDER_BAD_REQUEST (not retryable) ──────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_400_bad_request() -> None:
    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "model not found"}})
    )
    provider = make_provider()
    with pytest.raises(COALAIError) as exc_info:
        await provider.complete(make_request(), timeout_seconds=10.0)
    await provider.close()

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_BAD_REQUEST
    assert exc_info.value.retryable is False


# ── HTTP 401 → PROVIDER_AUTH_FAILURE (not retryable) ─────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_401_auth_failure() -> None:
    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
    )
    provider = make_provider()
    with pytest.raises(COALAIError) as exc_info:
        await provider.complete(make_request(), timeout_seconds=10.0)
    await provider.close()

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_AUTH_FAILURE
    assert exc_info.value.retryable is False


# ── Connection timeout → PROVIDER_TIMEOUT ─────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_timeout() -> None:
    import asyncio

    async def slow_response(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json=SUCCESSFUL_RESPONSE)

    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(side_effect=slow_response)
    provider = make_provider()
    with pytest.raises(COALAIError) as exc_info:
        await provider.complete(make_request(), timeout_seconds=0.05)
    await provider.close()

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_TIMEOUT
    assert exc_info.value.retryable is True


# ── Malformed response → PROVIDER_MALFORMED_RESPONSE ─────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_malformed_response() -> None:
    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": "structure"})
    )
    provider = make_provider()
    with pytest.raises(COALAIError) as exc_info:
        await provider.complete(make_request(), timeout_seconds=10.0)
    await provider.close()

    assert exc_info.value.error_type == COALAIErrorType.PROVIDER_MALFORMED_RESPONSE
    assert exc_info.value.retryable is False


# ── Streaming: chunks yielded ─────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_stream_chunks_yielded() -> None:
    sse_lines = "\n".join([
        'data: {"id":"c1","object":"chat.completion.chunk","model":"llama-3.1-8b-instant","choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}',
        'data: {"id":"c1","object":"chat.completion.chunk","model":"llama-3.1-8b-instant","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}',
        'data: {"id":"c1","object":"chat.completion.chunk","model":"llama-3.1-8b-instant","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "data: [DONE]",
    ])

    respx.post(f"{BASE_URL}/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=sse_lines,
            headers={"content-type": "text/event-stream"},
        )
    )

    provider = make_provider()
    chunks = []
    async for chunk in provider.stream(make_request(stream=True), timeout_seconds=10.0):
        chunks.append(chunk)
    await provider.close()

    content = "".join(c.content for c in chunks if c.content)
    assert content == "Hello"
    final = chunks[-1]
    assert final.finish_reason == FinishReason.STOP
