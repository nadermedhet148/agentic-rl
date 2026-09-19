"""Langfuse tracing — see docs/OBSERVABILITY.md for the full setup guide.

Design intent: every function here is safe to call whether or not tracing is
enabled, configured, or even installed. `Agent`, `ClaudePlanner`, and
`ClaudeDistiller` import and call this module unconditionally; nothing in this
project has a hard runtime dependency on the `langfuse` package. That's what
keeps 130+ existing tests network-free and deterministic without touching them —
`setup()` is only ever invoked from `api/app.py`, gated on
`Settings.observability_enabled` (default False).

Two kinds of Langfuse observation are used:
- `span()` — a generic unit of work (an Agent method, the consolidator).
- `generation()` — specifically an LLM call. This project calls the Claude API
  via `client.messages.parse()`, which posts directly rather than going through
  `.create()` — so the (otherwise very convenient) `opentelemetry-instrumentation-
  anthropic` auto-instrumentation does not see these calls. They're instrumented
  by hand instead, at the two call sites (llm/claude.py, llm/distiller.py).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse import Langfuse

    from agentic_rl.core.config import Settings

logger = logging.getLogger(__name__)

_client: Langfuse | None = None


def setup(settings: Settings) -> bool:
    """Activate Langfuse tracing if `Settings.observability_enabled` is set.

    Returns whether tracing actually activated. Never raises: a missing
    `langfuse` package (it's an optional extra — see pyproject.toml
    `[project.optional-dependencies] observability`) or missing credentials
    just leaves tracing off, logged once as a warning.
    """
    global _client
    if not settings.observability_enabled:
        return False

    try:
        from langfuse import get_client
    except ImportError as exc:
        logger.warning(
            "AGENTIC_RL_OBSERVABILITY_ENABLED is set but the `langfuse` package "
            "isn't installed. Run: pip install -e '.[observability]'. (%s)",
            exc,
        )
        return False

    _client = get_client()
    logger.info("Langfuse tracing enabled — see docs/OBSERVABILITY.md to verify credentials.")
    return True


def enabled() -> bool:
    return _client is not None


@contextmanager
def span(name: str, *, input: Any = None, **metadata: Any) -> Iterator[Any]:
    """A generic Langfuse span, or a true no-op (yields None) when tracing isn't
    active — safe to wrap around any block of code unconditionally."""
    if _client is None:
        yield None
        return
    with _client.start_as_current_observation(
        as_type="span", name=name, input=input, metadata=metadata or None
    ) as observation:
        yield observation


@contextmanager
def generation(name: str, *, model: str, input: Any = None) -> Iterator[Any]:
    """A Langfuse generation wrapping one LLM call. See module docstring for why
    this is manual rather than auto-instrumented."""
    if _client is None:
        yield None
        return
    with _client.start_as_current_observation(
        as_type="generation", name=name, model=model, input=input
    ) as observation:
        yield observation


def update_current_span(**fields: Any) -> None:
    """No-op when tracing isn't active. `fields` match Langfuse's
    `update_current_span` kwargs: output, metadata, level, status_message, ..."""
    if _client is not None:
        _client.update_current_span(**fields)


def update_current_generation(**fields: Any) -> None:
    """No-op when tracing isn't active. `fields` match Langfuse's
    `update_current_generation` kwargs: output, usage_details, level, ..."""
    if _client is not None:
        _client.update_current_generation(**fields)


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    # Anthropic's raw Message.usage field names:
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    # PydanticAI's RunUsage field names (see llm/claude.py, llm/distiller.py —
    # both pass a PydanticAI AgentRunResult here, whose `.usage` is a RunUsage):
    "cache_write_tokens",
    "cache_read_tokens",
)


def usage_details(response: Any) -> dict[str, int] | None:
    """Extract token counts from something with a `.usage` attribute — either an
    Anthropic Message response or a PydanticAI AgentRunResult — into the
    `Dict[str, int]` shape Langfuse's `usage_details` expects. Defensive: usage
    field names differ between the two and have grown over time (cache read/
    creation tokens, ...), so this picks up whatever's present."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details = {}
    for field in _USAGE_FIELDS:
        value = getattr(usage, field, None)
        if isinstance(value, int):
            details[field] = value
    return details or None


def flush() -> None:
    """Force-send any buffered traces. Call on app shutdown — see api/app.py."""
    if _client is not None:
        _client.flush()


def shutdown() -> None:
    if _client is not None:
        _client.shutdown()
