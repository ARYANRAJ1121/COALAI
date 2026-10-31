"""Startup wiring tests for the default provider fallback route."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from coalai import main


class StubProvider:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.close = AsyncMock()


@pytest.mark.asyncio
async def test_configured_default_model_uses_default_fallback_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit default-model requests resolve to the same chain as `default`."""
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "init_redis", lambda: None)
    monkeypatch.setattr(main, "close_db", AsyncMock())
    monkeypatch.setattr(main, "close_redis", AsyncMock())
    monkeypatch.setattr(main.settings, "groq_enabled", True)
    monkeypatch.setattr(main.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(main.settings, "openrouter_enabled", True)
    monkeypatch.setattr(main.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(main, "OllamaProvider", lambda **_: StubProvider("ollama"))
    monkeypatch.setattr(main, "GroqProvider", lambda **_: StubProvider("groq"))
    monkeypatch.setattr(main, "OpenRouterProvider", lambda **_: StubProvider("openrouter"))

    app = main.create_app()
    async with app.router.lifespan_context(app):
        registry = app.state.provider_registry
        default_targets = registry.resolve_targets("default")
        explicit_default_targets = registry.resolve_targets(main.settings.ollama_default_model)

        assert explicit_default_targets == default_targets
        assert [target.provider_id for target in explicit_default_targets] == [
            "ollama",
            "groq",
            "openrouter",
        ]
