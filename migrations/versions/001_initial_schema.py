"""
Initial schema — Milestone 1

Creates: tenants, api_keys, model_registry, accounting_ledger, budgets

Revision: 001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── tenants ───────────────────────────────────────────────────────────────
    op.create_table(
        "tenants",
        sa.Column("tenant_id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # ── api_keys ──────────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("key_id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requests_per_minute", sa.Integer, nullable=False, server_default="60"),
        sa.Column("requests_per_day", sa.Integer, nullable=False, server_default="10000"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])

    # ── model_registry ────────────────────────────────────────────────────────
    op.create_table(
        "model_registry",
        sa.Column("model_id", sa.String(255), primary_key=True, nullable=False),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("context_window", sa.Integer, nullable=False),
        sa.Column("supports_tools", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("supports_streaming", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("input_cost_per_1k_tokens", sa.Numeric(10, 8), nullable=False, server_default="0"),
        sa.Column("output_cost_per_1k_tokens", sa.Numeric(10, 8), nullable=False, server_default="0"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # ── accounting_ledger ─────────────────────────────────────────────────────
    op.create_table(
        "accounting_ledger",
        sa.Column("event_id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("request_id", UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=True),
        sa.Column("provider", sa.String(100), nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Numeric(12, 8), nullable=False, server_default="0"),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("trace_id", sa.String(128), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="CONFIRMED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("request_id", name="uq_accounting_ledger_request_id"),
    )
    op.create_index("ix_accounting_ledger_tenant_id", "accounting_ledger", ["tenant_id"])
    op.create_index("ix_accounting_ledger_status", "accounting_ledger", ["status"])
    op.create_index("ix_accounting_ledger_created_at", "accounting_ledger", ["created_at"])

    # ── budgets ───────────────────────────────────────────────────────────────
    # Schema present from Milestone 1. NOT enforced until Phase 7.
    op.create_table(
        "budgets",
        sa.Column("budget_id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("period", sa.String(20), nullable=False, server_default="MONTHLY"),
        sa.Column("limit_usd", sa.Numeric(12, 4), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"], ondelete="CASCADE"),
    )

    # ── Seed: default Ollama models ───────────────────────────────────────────
    op.execute("""
        INSERT INTO model_registry (model_id, provider, display_name, context_window,
            supports_streaming, input_cost_per_1k_tokens, output_cost_per_1k_tokens)
        VALUES
            ('llama3.2:3b',   'ollama', 'Llama 3.2 3B',   128000, true, 0, 0),
            ('llama3.2:1b',   'ollama', 'Llama 3.2 1B',   128000, true, 0, 0),
            ('qwen2.5:0.5b',  'ollama', 'Qwen 2.5 0.5B',  32768,  true, 0, 0),
            ('qwen2.5:1.5b',  'ollama', 'Qwen 2.5 1.5B',  32768,  true, 0, 0),
            ('mistral:7b',    'ollama', 'Mistral 7B',      32768,  true, 0, 0)
        ON CONFLICT (model_id) DO NOTHING;
    """)


def downgrade() -> None:
    op.drop_table("budgets")
    op.drop_index("ix_accounting_ledger_created_at", "accounting_ledger")
    op.drop_index("ix_accounting_ledger_status", "accounting_ledger")
    op.drop_index("ix_accounting_ledger_tenant_id", "accounting_ledger")
    op.drop_table("accounting_ledger")
    op.drop_table("model_registry")
    op.drop_index("ix_api_keys_key_hash", "api_keys")
    op.drop_table("api_keys")
    op.drop_table("tenants")
