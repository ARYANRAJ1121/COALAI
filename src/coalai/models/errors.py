"""
COALAI error taxonomy.

All errors raised within COALAI are instances of COALAIError.
Provider-specific errors are normalized to these types before leaving
the provider adapter. This means the reliability engine and gateway
never need to know which specific provider raised an exception.
"""

from __future__ import annotations

from enum import StrEnum


class COALAIErrorType(StrEnum):
    # ── Provider errors ───────────────────────────────────────────────────────
    # These are surfaced after all retry/fallback attempts are exhausted.
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_SERVER_ERROR = "provider_server_error"
    PROVIDER_AUTH_FAILURE = "provider_auth_failure"
    PROVIDER_BAD_REQUEST = "provider_bad_request"
    PROVIDER_MALFORMED_RESPONSE = "provider_malformed_response"
    ALL_PROVIDERS_FAILED = "all_providers_failed"

    # ── Gateway / client errors ───────────────────────────────────────────────
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"

    # ── Infrastructure errors ─────────────────────────────────────────────────
    # 503 — returned to the client when a critical internal operation fails
    # (e.g., accounting write — see v0.3 Option A accounting invariant).
    ACCOUNTING_WRITE_FAILED = "accounting_write_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"


# Maps error type → default HTTP status code.
# The gateway uses this to produce the correct HTTP response.
_DEFAULT_HTTP_STATUS: dict[COALAIErrorType, int] = {
    COALAIErrorType.PROVIDER_TIMEOUT: 504,
    COALAIErrorType.PROVIDER_RATE_LIMITED: 429,
    COALAIErrorType.PROVIDER_SERVER_ERROR: 502,
    COALAIErrorType.PROVIDER_AUTH_FAILURE: 502,
    COALAIErrorType.PROVIDER_BAD_REQUEST: 400,
    COALAIErrorType.PROVIDER_MALFORMED_RESPONSE: 502,
    COALAIErrorType.ALL_PROVIDERS_FAILED: 503,
    COALAIErrorType.AUTHENTICATION_FAILED: 401,
    COALAIErrorType.AUTHORIZATION_FAILED: 403,
    COALAIErrorType.VALIDATION_ERROR: 422,
    COALAIErrorType.RATE_LIMITED: 429,
    COALAIErrorType.INVALID_REQUEST: 400,
    COALAIErrorType.ACCOUNTING_WRITE_FAILED: 503,
    COALAIErrorType.SERVICE_UNAVAILABLE: 503,
}

# Whether each error type is safe to retry on a different provider.
_RETRYABLE: dict[COALAIErrorType, bool] = {
    COALAIErrorType.PROVIDER_TIMEOUT: True,
    COALAIErrorType.PROVIDER_RATE_LIMITED: True,
    COALAIErrorType.PROVIDER_SERVER_ERROR: True,
    COALAIErrorType.PROVIDER_AUTH_FAILURE: False,
    COALAIErrorType.PROVIDER_BAD_REQUEST: False,
    COALAIErrorType.PROVIDER_MALFORMED_RESPONSE: False,
    COALAIErrorType.ALL_PROVIDERS_FAILED: False,
    COALAIErrorType.AUTHENTICATION_FAILED: False,
    COALAIErrorType.AUTHORIZATION_FAILED: False,
    COALAIErrorType.VALIDATION_ERROR: False,
    COALAIErrorType.RATE_LIMITED: False,
    COALAIErrorType.INVALID_REQUEST: False,
    COALAIErrorType.ACCOUNTING_WRITE_FAILED: False,
    COALAIErrorType.SERVICE_UNAVAILABLE: False,
}


class COALAIError(Exception):
    """
    Single exception type raised throughout the COALAI codebase.

    Provider adapters catch their own exceptions and raise COALAIError.
    The gateway catches COALAIError and converts it to an HTTP response.
    No other exception type crosses module boundaries.
    """

    def __init__(
        self,
        error_type: COALAIErrorType,
        message: str,
        *,
        retryable: bool | None = None,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retryable = retryable if retryable is not None else _RETRYABLE[error_type]
        self.http_status = http_status if http_status is not None else _DEFAULT_HTTP_STATUS[error_type]
        # Populated when the provider returns a Retry-After header.
        self.retry_after_seconds = retry_after_seconds

    def __repr__(self) -> str:
        return f"COALAIError(type={self.error_type}, message={self.message!r}, retryable={self.retryable})"
