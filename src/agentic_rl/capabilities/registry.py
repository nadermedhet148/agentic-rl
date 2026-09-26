from __future__ import annotations

from agentic_rl.capabilities.base import Capability


class CapabilityRegistry:
    """Holds the capabilities available to the planner and executor."""

    def __init__(self) -> None:
        self._by_name: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if capability.name in self._by_name:
            raise ValueError(f"capability already registered: {capability.name}")
        self._by_name[capability.name] = capability

    def get(self, name: str) -> Capability:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown capability: {name}") from None

    def get_or_none(self, name: str) -> Capability | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return list(self._by_name)

    def view(self, names: list[str]) -> CapabilityRegistry:
        """A registry holding only `names`, sharing the same capability instances —
        one agent's slice of the team's capabilities (core/models.py AgentProfile)."""
        subset = CapabilityRegistry()
        for name in names:
            if name not in subset._by_name:
                subset.register(self.get(name))
        return subset

    def tool_schemas(self) -> list[dict]:
        """JSON-schema tool definitions for the planner, one per registered capability."""
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "input_schema": cap.input_schema,
            }
            for cap in self._by_name.values()
        ]
