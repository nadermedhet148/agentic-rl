from __future__ import annotations

import asyncio
from typing import Any, Protocol

from agentic_rl.capabilities.base import Capability, Outcome, Tier

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {"type": "integer", "description": "Max results to return."},
    },
    "required": ["query"],
    "additionalProperties": False,
}


class SearchPort(Protocol):
    """What WebSearchCapability needs from the search backend.

    Kept as a narrow protocol (rather than depending on ddgs directly in the
    capability) so it can be unit-tested with a fake instead of hitting the
    network, the same way ScheduleTaskCapability uses SchedulerPort.
    """

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Run a search, returning raw result dicts (ddgs' own shape)."""
        ...


class DdgsSearchPort:
    """Real SearchPort backed by the `ddgs` (DuckDuckGo search) library."""

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))


class WebSearchCapability(Capability):
    """Searches the web and returns matching page titles, URLs, and snippets."""

    name = "web_search"
    description = "Search the web for a query and return matching page titles, URLs, and snippets."
    input_schema = INPUT_SCHEMA

    def __init__(self, search: SearchPort, max_results: int = 5, timeout_s: float = 10.0):
        self._search = search
        self._max_results = max_results
        self._timeout_s = timeout_s

    def tier_for(self, params: dict[str, Any]) -> Tier:
        return Tier.READ

    async def execute(self, params: dict[str, Any]) -> Outcome:
        query = params.get("query")
        if not query:
            return Outcome(ok=False, error="query is required")
        max_results = int(params.get("max_results") or self._max_results)

        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._search.text, str(query), max_results),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            return Outcome(ok=False, error="web_search timed out")
        except Exception as exc:  # noqa: BLE001 - surfaced as a normal failed outcome
            return Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
        return Outcome(ok=True, status="200", payload={"results": results})
