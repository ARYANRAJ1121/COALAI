"""
SQLAlchemy ORM models — Milestone 1 schema.

All tables include tenant_id. This is enforced at the application layer.
PostgreSQL row-level security is added in a later security hardening phase.

Naming conventions:
  - Table names are snake_case plural nouns.
  - Primary keys are UUIDs (gen_random_uuid() server-side default).
  - Timestamps use TIMESTAMPTZ (UTC stored, timezone-aware on retrieval).
  - JSONB used for flexible structured fields (configs, metadata).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
    )
    # sha256(raw_key) — the raw key is never stored.
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # Human-readable label (e.g., "ci-runner", "dev-laptop").
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Requests per minute / per day limits for this key.
    requests_per_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    requests_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=10000)

    __table_args__ = (Index("ix_api_keys_key_hash", "key_hash"),)


class ModelRegistry(Base):
    """
    Canonical list of LLM models known to COALAI.
    The router reads from this table (via in-process cache) to make decisions.
    """

    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    # Human-readable name shown in API responses.
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Phase 7: cost per 1k tokens (USD). 0.0 for local/free models.
    input_cost_per_1k_tokens: Mapped[float] = mapped_column(Numeric(10, 8), nullable=False, default=0.0)
    output_cost_per_1k_tokens: Mapped[float] = mapped_column(Numeric(10, 8), nullable=False, default=0.0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AccountingLedger(Base):
    """
    Durable record of every completed COALAI request.

    v0.3 Option A accounting invariant:
      - Non-streaming: status='CONFIRMED' written BEFORE the HTTP response.
      - Streaming: status='PENDING' written BEFORE the first SSE chunk,
        then updated to 'CONFIRMED' after the stream ends.

    A request that returns HTTP 2xx ALWAYS has a row here.
    A row with status='PENDING' older than accounting_pending_ttl_minutes
    is an anomaly — the reconciliation job will promote it.
    """

    __tablename__ = "accounting_ledger"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id", ondelete="RESTRICT"),
        nullable=False,
    )
    model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 0.0 for local/free models. Populated for future cost visibility.
    estimated_cost_usd: Mapped[float] = mapped_column(Numeric(12, 8), nullable=False, default=0.0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 'PENDING' (streaming, not yet complete) or 'CONFIRMED' (fully recorded).
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="CONFIRMED")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Enforce idempotency: one accounting row per COALAI request.
        UniqueConstraint("request_id", name="uq_accounting_ledger_request_id"),
        Index("ix_accounting_ledger_tenant_id", "tenant_id"),
        Index("ix_accounting_ledger_status", "status"),
        Index("ix_accounting_ledger_created_at", "created_at"),
    )


class Budget(Base):
    """
    Per-tenant budget limits. Schema present from Milestone 1.
    NOT enforced until Phase 7.
    """

    __tablename__ = "budgets"

    budget_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    # 'MONTHLY' or 'DAILY'
    period: Mapped[str] = mapped_column(String(20), nullable=False, default="MONTHLY")
    limit_usd: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
