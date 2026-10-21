"""
COALAI FastAPI application entry point.

Responsibilities:
  - Application factory
  - Startup/shutdown lifespan (DB, Redis, provider initialization)
  - Exception handler registration
  - Router registration

All long-lived resources (DB engine, Redis client, Ollama provider) are
initialized in the lifespan context and stored in app.state so they are
accessible from route handlers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from coalai.cache.redis_client import close_redis, init_redis
from coalai.config import settings
from coalai.db.session import close_db, init_db
from coalai.gateway.router import router
from coalai.gateway.schemas import ErrorDetail, ErrorResponse
from coalai.models.errors import COALAIError
from coalai.observability.logging import configure_logging, get_logger
from coalai.providers.ollama import OllamaProvider
from coalai.providers.registry import ProviderRegistry
from coalai.reliability.circuit_breaker import CircuitBreaker
from coalai.reliability.retry import RetryPolicy
from coalai.reliability.routing import FallbackRouter

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    All initialization happens here so it runs before any request is handled.
    Teardown happens on shutdown.
    """
    configure_logging(debug=settings.debug)
    log.info("coalai_starting", version=settings.app_version, debug=settings.debug)

    # Initialize infrastructure
    init_db()
    init_redis()

    # Create provider singleton, circuit breaker, registry, and router
    provider_registry = ProviderRegistry()
    circuit_breaker = CircuitBreaker()
    retry_policy = RetryPolicy.from_settings()

    ollama = OllamaProvider(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
    )
    provider_registry.register_provider(ollama)

    fallback_router = FallbackRouter(
        registry=provider_registry,
        circuit_breaker=circuit_breaker,
        retry_policy=retry_policy,
    )

    app.state.provider_registry = provider_registry
    app.state.circuit_breaker = circuit_breaker
    app.state.retry_policy = retry_policy
    app.state.ollama_provider = ollama
    app.state.fallback_router = fallback_router

    log.info(
        "coalai_started",
        ollama_base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
    )

    yield  # ← Application is running

    # Teardown
    log.info("coalai_shutting_down")
    await app.state.ollama_provider.close()
    await close_db()
    await close_redis()
    log.info("coalai_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="COALAI",
        description="Cost-Optimized AI Infrastructure — Production-grade AI execution infrastructure for reliable, cost-aware LLM and agent workloads.",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)

    # Global exception handlers
    @app.exception_handler(COALAIError)
    async def coalai_error_handler(request: Request, exc: COALAIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                error=ErrorDetail(
                    error_type=exc.error_type,
                    message=exc.message,
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    error_type="internal_error",
                    message="An unexpected error occurred.",
                )
            ).model_dump(),
        )

    return app


# ASGI application instance used by uvicorn
app = create_app()
