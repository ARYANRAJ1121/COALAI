"""
Async SQLAlchemy engine and session management.

We use SQLAlchemy 2.0 async mode with asyncpg throughout.
Retrofitting sync → async is expensive; starting async avoids that cost.

Usage in application code:
    async with get_session() as session:
        result = await session.execute(select(Tenant))

Usage in FastAPI via dependency injection:
    async def endpoint(session: AsyncSession = Depends(get_db_session)):
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from coalai.config import settings
from coalai.observability.logging import get_logger

log = get_logger(__name__)

# Module-level engine. Created once at application startup.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the module-level async engine. Raises if not initialised."""
    if _engine is None:
        raise RuntimeError(
            "Database engine is not initialised. "
            "Call init_db() during application startup."
        )
    return _engine


def init_db() -> None:
    """
    Create the async engine and session factory.
    Must be called once at application startup before any DB access.
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        pool_pre_ping=True,  # Detect stale connections before use
        echo=settings.debug,  # Log SQL in debug mode
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    log.info("database_engine_initialised", pool_size=settings.db_pool_size)


async def close_db() -> None:
    """Dispose the engine pool. Call during application shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        log.info("database_engine_closed")
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session.
    Commits on success, rolls back on any exception.

    Use this for code outside FastAPI's dependency injection (e.g., CLI scripts).
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.
    Transaction is committed on successful response, rolled back on error.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
