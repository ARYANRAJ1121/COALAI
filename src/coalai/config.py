from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All COALAI configuration is sourced from environment variables.
    No configuration lives in code. No defaults expose credentials.

    Load order: environment variables → .env file → field defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "COALAI"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://coalai:coalai@localhost:5432/coalai"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout_seconds: float = 30.0

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    # Any single Redis operation that exceeds this is abandoned (fail-open).
    redis_timeout_seconds: float = 0.5

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    # Default model used when the client does not specify one.
    ollama_default_model: str = "llama3.2:3b"

    # ── Groq (free tier cloud inference) ──────────────────────────────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com"
    groq_default_model: str = "llama-3.1-8b-instant"
    groq_enabled: bool = False

    # ── OpenRouter (free model aggregator) ────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai"
    openrouter_default_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_enabled: bool = False

    # ── Reliability ───────────────────────────────────────────────────────────
    provider_timeout_seconds: float = 30.0
    provider_streaming_timeout_seconds: float = 300.0
    retry_max_attempts: int = 3
    retry_initial_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    # Fraction of computed delay used as random jitter band (±).
    retry_jitter_factor: float = 0.25

    # ── Circuit Breaker ───────────────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_time_seconds: float = 30.0
    circuit_breaker_half_open_success_threshold: int = 2
    circuit_breaker_half_open_max_trials: int = 2
    circuit_breaker_canary_timeout_seconds: float = 10.0

    # ── Auth ──────────────────────────────────────────────────────────────────
    # How long a verified TenantContext is cached in Redis before re-validation.
    api_key_cache_ttl_seconds: int = 300

    # ── Accounting ────────────────────────────────────────────────────────────
    # PENDING accounting rows older than this window are anomalies.
    accounting_pending_ttl_minutes: int = 10

    @field_validator("retry_jitter_factor")
    @classmethod
    def jitter_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("retry_jitter_factor must be between 0.0 and 1.0")
        return v

    @field_validator("db_pool_size")
    @classmethod
    def pool_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("db_pool_size must be >= 1")
        return v


# Module-level singleton. All application code imports from here.
settings = Settings()
