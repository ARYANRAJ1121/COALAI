"""
Structured logging setup for COALAI.

All logs are emitted as JSON to stdout. No rotating files — log collection
is the responsibility of the container orchestrator (Docker, Kubernetes).

Rules enforced here:
  - Every log entry has: timestamp, level, request_id, trace_id, event.
  - tenant_id is included ONLY when it has been verified by the auth module.
    It is NEVER sourced from untrusted request data.
  - No f-string log messages. Every dynamic value is a keyword argument.

Usage:
    from coalai.observability.logging import get_logger
    log = get_logger(__name__)
    log.info("provider_call_completed", provider="ollama", latency_ms=304)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(debug: bool = False) -> None:
    """
    Configure structlog for JSON output to stdout.
    Call once at application startup (in main.py lifespan).
    """
    level = logging.DEBUG if debug else logging.INFO

    # Configure the stdlib logging backend that structlog writes to.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)


def bind_request_context(
    request_id: str,
    trace_id: str,
    *,
    tenant_id: str | None = None,
) -> None:
    """
    Bind request-scoped fields to the structlog context.
    These fields will appear in every log entry for this request/task.

    tenant_id MUST be None until the auth module has verified the API key.
    Pass tenant_id only after TenantContext is populated.
    """
    ctx: dict[str, Any] = {"request_id": request_id, "trace_id": trace_id}
    if tenant_id is not None:
        ctx["tenant_id"] = tenant_id
    structlog.contextvars.bind_contextvars(**ctx)


def clear_request_context() -> None:
    """Clear all request-scoped context. Call at the end of each request."""
    structlog.contextvars.clear_contextvars()
