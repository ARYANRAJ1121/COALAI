"""
OpenRouter provider adapter.

OpenRouter (https://openrouter.ai) is a unified API gateway that aggregates
multiple LLM providers. It offers several completely free models including:
  - meta-llama/llama-3.1-8b-instruct:free
  - mistralai/mistral-7b-instruct:free
  - google/gemma-2-9b-it:free

The API is OpenAI-compatible at https://openrouter.ai/api/v1.

Error normalization table:
  HTTP 408 / asyncio.TimeoutError  → PROVIDER_TIMEOUT      (retryable)
  HTTP 429                         → PROVIDER_RATE_LIMITED  (retryable)
  HTTP 5xx                         → PROVIDER_SERVER_ERROR  (retryable)
  HTTP 401 / 403                   → PROVIDER_AUTH_FAILURE  (not retryable)
  HTTP 400 / 422                   → PROVIDER_BAD_REQUEST   (not retryable)
  Malformed / non-JSON response    → PROVIDER_MALFORMED_RESPONSE (not retryable)
  Connection error                 → PROVIDER_SERVER_ERROR  (retryable)

No httpx exception type leaves this module. Every exception is caught
and re-raised as COALAIError before returning to the caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx

from coalai.config import settings
from coalai.models.contracts import (
    ExecutionResult,
    FinishReason,
    Message,
    NormalizedChunk,
    NormalizedRequest,
    utcnow,
)
from coalai.models.errors import COALAIError, COALAIErrorType
from coalai.observability.logging import get_logger
from coalai.providers.base import LLMProvider, ProviderHealth, ProviderStatus

log = get_logger(__name__)

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALL,
    "content_filter": FinishReason.STOP,
}


def _map_finish_reason(raw: str | None) -> FinishReason:
    if raw is None:
        return FinishReason.STOP
    return _FINISH_REASON_MAP.get(raw.lower(), FinishReason.STOP)


def _normalize_http_error(response: httpx.Response, provider_name: str) -> COALAIError:
    """Convert an HTTP error response to a COALAIError."""
    status = response.status_code
    try:
        body = response.json()
        detail = body.get("error", {}).get("message", response.text[:200])
    except Exception:
        detail = response.text[:200]

    if status == 429:
        retry_after: float | None = None
        if "retry-after" in response.headers:
            try:
                retry_after = float(response.headers["retry-after"])
            except ValueError:
                pass
        return COALAIError(
            COALAIErrorType.PROVIDER_RATE_LIMITED,
            f"{provider_name} rate limited: {detail}",
            retry_after_seconds=retry_after,
        )
    if status in (401, 403):
        return COALAIError(
            COALAIErrorType.PROVIDER_AUTH_FAILURE,
            f"{provider_name} auth failure (HTTP {status}): {detail}",
        )
    if status in (400, 422):
        return COALAIError(
            COALAIErrorType.PROVIDER_BAD_REQUEST,
            f"{provider_name} rejected request (HTTP {status}): {detail}",
        )
    if status >= 500:
        return COALAIError(
            COALAIErrorType.PROVIDER_SERVER_ERROR,
            f"{provider_name} server error (HTTP {status}): {detail}",
        )
    return COALAIError(
        COALAIErrorType.PROVIDER_SERVER_ERROR,
        f"Unexpected {provider_name} response (HTTP {status}): {detail}",
    )


class OpenRouterProvider:
    """
    Adapter for OpenRouter's OpenAI-compatible chat completion API.

    OpenRouter aggregates many LLM providers behind a single API and offers
    several completely free open-source models. It serves as an excellent
    emergency fallback tier since it is free and has broad model coverage.

    Requires OPENROUTER_API_KEY environment variable to be set.
    """

    provider_id: str = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.openrouter_api_key
        self._base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self._default_model = default_model or settings.openrouter_default_model

        if not self._api_key:
            log.warning(
                "openrouter_no_api_key",
                msg="OPENROUTER_API_KEY not set — OpenRouter provider will fail on all requests",
            )

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=None,  # Timeouts are managed by our timeout wrapper
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                # OpenRouter recommends identifying the calling app
                "HTTP-Referer": "https://github.com/ARYANRAJ1121/COALAI",
                "X-Title": "COALAI Gateway",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _build_payload(self, request: NormalizedRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": request.stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    async def complete(
        self,
        request: NormalizedRequest,
        *,
        timeout_seconds: float,
    ) -> ExecutionResult:
        """Non-streaming chat completion via OpenRouter."""
        started_at = utcnow()
        payload = self._build_payload(request)

        try:
            response = await asyncio.wait_for(
                self._client.post("/api/v1/chat/completions", json=payload),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            log.warning(
                "openrouter_timeout",
                model=request.model,
                timeout_seconds=timeout_seconds,
            )
            raise COALAIError(
                COALAIErrorType.PROVIDER_TIMEOUT,
                f"OpenRouter did not respond within {timeout_seconds}s",
            )
        except httpx.ConnectError as exc:
            raise COALAIError(
                COALAIErrorType.PROVIDER_SERVER_ERROR,
                f"OpenRouter connection failed: {exc}",
            )
        except httpx.RequestError as exc:
            raise COALAIError(
                COALAIErrorType.PROVIDER_SERVER_ERROR,
                f"OpenRouter request error: {exc}",
            )

        completed_at = utcnow()

        if response.status_code != 200:
            raise _normalize_http_error(response, "OpenRouter")

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = _map_finish_reason(choice.get("finish_reason"))
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            provider_request_id = data.get("id")
        except (KeyError, IndexError, ValueError) as exc:
            raise COALAIError(
                COALAIErrorType.PROVIDER_MALFORMED_RESPONSE,
                f"OpenRouter returned malformed response: {exc}",
            )

        log.info(
            "openrouter_complete_success",
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round((completed_at - started_at).total_seconds() * 1000, 1),
        )

        return ExecutionResult(
            success=True,
            content=content,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_used=self.provider_id,
            model_used=request.model,
            provider_request_id=provider_request_id,
            execution_started_at=started_at,
            execution_completed_at=completed_at,
            first_token_at=None,
            raw_provider_response=data,
        )

    async def stream(
        self,
        request: NormalizedRequest,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[NormalizedChunk]:
        """Streaming chat completion via OpenRouter SSE."""
        started_at = utcnow()
        payload = self._build_payload(request)
        first_token_at: datetime | None = None
        input_tokens = 0
        output_tokens = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                async with self._client.stream(
                    "POST", "/api/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        raise _normalize_http_error(response, "OpenRouter")

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            chunk_data = __import__("json").loads(line)
                        except ValueError:
                            continue

                        choice = chunk_data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        chunk_content = delta.get("content") or ""
                        raw_finish = choice.get("finish_reason")
                        finish = _map_finish_reason(raw_finish) if raw_finish else None

                        if chunk_content and first_token_at is None:
                            first_token_at = utcnow()

                        usage = chunk_data.get("usage", {})
                        if usage:
                            input_tokens = usage.get("prompt_tokens", input_tokens)
                            output_tokens = usage.get("completion_tokens", output_tokens)

                        yield NormalizedChunk(
                            content=chunk_content,
                            finish_reason=finish,
                            input_tokens=input_tokens if finish else None,
                            output_tokens=output_tokens if finish else None,
                        )

        except asyncio.TimeoutError:
            log.warning(
                "openrouter_stream_timeout",
                model=request.model,
                timeout_seconds=timeout_seconds,
            )
            raise COALAIError(
                COALAIErrorType.PROVIDER_TIMEOUT,
                f"OpenRouter stream timed out after {timeout_seconds}s",
            )
        except httpx.ConnectError as exc:
            raise COALAIError(
                COALAIErrorType.PROVIDER_SERVER_ERROR,
                f"OpenRouter connection failed during stream: {exc}",
            )
        except httpx.RequestError as exc:
            raise COALAIError(
                COALAIErrorType.PROVIDER_SERVER_ERROR,
                f"OpenRouter stream request error: {exc}",
            )
        except COALAIError:
            raise  # Already normalized

    async def health_check(self) -> ProviderHealth:
        """Check if OpenRouter API is reachable by listing models."""
        started_at = utcnow()
        try:
            response = await asyncio.wait_for(
                self._client.get("/api/v1/models"),
                timeout=5.0,
            )
            latency_ms = (utcnow() - started_at).total_seconds() * 1000
            if response.status_code == 200:
                return ProviderHealth(
                    provider_id=self.provider_id,
                    status=ProviderStatus.HEALTHY,
                    latency_ms=round(latency_ms, 1),
                )
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderStatus.DEGRADED,
                latency_ms=round(latency_ms, 1),
                error=f"HTTP {response.status_code}",
            )
        except Exception as exc:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderStatus.UNREACHABLE,
                latency_ms=None,
                error=str(exc),
            )
