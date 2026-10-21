"""
Provider Registry and Route Target Definitions.

Manages:
  - Registered LLMProvider instances.
  - Model alias routing mappings (e.g. "default" or "llama3" → [primary, fallback]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from coalai.config import settings
from coalai.providers.base import LLMProvider


@dataclass(frozen=True)
class RouteTarget:
    """Target provider and model pair."""
    provider_id: str
    model: str

    @property
    def key(self) -> str:
        """Key used for circuit breaking and metrics."""
        return f"{self.provider_id}:{self.model}"


@dataclass
class ModelRoute:
    """Route definition consisting of an ordered list of targets."""
    model_alias: str
    targets: list[RouteTarget] = field(default_factory=list)


class ProviderRegistry:
    """Registry for active providers and model routing paths."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._routes: dict[str, list[RouteTarget]] = {}

    def register_provider(self, provider: LLMProvider) -> None:
        """Register an LLMProvider implementation."""
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: str) -> LLMProvider | None:
        """Retrieve a provider by its identifier."""
        return self._providers.get(provider_id)

    def register_route(self, alias: str, targets: list[RouteTarget]) -> None:
        """Register an ordered chain of targets for a model alias."""
        self._routes[alias] = targets

    def resolve_targets(self, model_hint: str | None) -> list[RouteTarget]:
        """
        Resolve an incoming model request to an ordered list of targets.
        If the model matches a registered alias, return its targets.
        Otherwise, map to primary provider with requested or default model.
        """
        requested = model_hint or settings.ollama_default_model

        if requested in self._routes:
            return list(self._routes[requested])

        # If "default" alias exists and model_hint was empty/default
        if not model_hint and "default" in self._routes:
            return list(self._routes["default"])

        # Default single-target fallback
        return [RouteTarget(provider_id="ollama", model=requested)]
