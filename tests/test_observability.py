from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic_rl.core import observability
from agentic_rl.core.config import Settings


@pytest.fixture(autouse=True)
def reset_client():
    """observability._client is a module-level singleton — isolate tests from
    each other and from whatever a previous test in the suite left behind."""
    original = observability._client
    observability._client = None
    yield
    observability._client = original


# --- disabled by default -------------------------------------------------------


def test_setup_returns_false_when_disabled():
    assert observability.setup(Settings(observability_enabled=False)) is False
    assert observability.enabled() is False


def test_trace_is_a_true_noop_when_disabled():
    with observability.trace("x", input="y", session_id="s1") as t:
        assert t is None


def test_span_is_a_true_noop_when_disabled():
    with observability.span("x", input="y", foo="bar") as s:
        assert s is None


def test_generation_is_a_true_noop_when_disabled():
    with observability.generation("x", model="claude-opus-5", input="y") as g:
        assert g is None


def test_get_current_trace_id_returns_none_when_disabled():
    assert observability.get_current_trace_id() is None


def test_update_and_flush_are_safe_noops_when_disabled():
    observability.update_current_span(output={"a": 1})  # must not raise
    observability.update_current_generation(output={"a": 1})  # must not raise
    observability.flush()  # must not raise
    observability.shutdown()  # must not raise


def test_usage_details_none_when_no_usage_attr():
    assert observability.usage_details(SimpleNamespace()) is None


def test_usage_details_extracts_int_fields():
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=None)
    response = SimpleNamespace(usage=usage)
    assert observability.usage_details(response) == {"input_tokens": 10, "output_tokens": 5}


# --- missing package -------------------------------------------------------


def test_setup_warns_and_stays_disabled_when_langfuse_not_installed():
    with patch.dict(sys.modules, {"langfuse": None}):
        result = observability.setup(Settings(observability_enabled=True))
    assert result is False
    assert observability.enabled() is False


# --- real SDK, tracing disabled (no network) ------------------------------


def test_setup_activates_and_spans_work_against_real_sdk(monkeypatch):
    # requires the `observability` extra (pip install -e ".[observability]") —
    # skip rather than fail in environments that only installed `[dev]`.
    pytest.importorskip("langfuse")
    # tracing_enabled must be the capitalized string "False" — a known quirk of
    # this SDK version's env-var parsing (see docs/OBSERVABILITY.md).
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-observability")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-observability")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "False")

    activated = observability.setup(Settings(observability_enabled=True))
    assert activated is True
    assert observability.enabled() is True

    # trace > span > generation nesting with shared trace_id across separate calls
    custom_trace_id = "46042ff0ff2d4a6eafa95e5985164b2a"

    # 1. api.chat creates the trace with trace_id
    with observability.trace("api.chat", input="do a thing", trace_id=custom_trace_id, session_id=custom_trace_id) as t:
        assert t is not None
        assert observability.get_current_trace_id() == custom_trace_id
        with observability.span("agent.run", input="do a thing", source="user") as span:
            assert span is not None
            assert observability.get_current_trace_id() == custom_trace_id
            with observability.generation("claude.plan", model="claude-opus-5", input="prompt") as gen:
                assert gen is not None
                assert observability.get_current_trace_id() == custom_trace_id
                observability.update_current_generation(output={"candidates": []}, usage_details={"input_tokens": 5})
            observability.update_current_span(output={"episode_id": custom_trace_id, "status": "executed"})

    # 2. api.feedback attaches to the same trace_id
    with observability.trace("api.feedback", input=custom_trace_id, trace_id=custom_trace_id, session_id=custom_trace_id) as t2:
        assert t2 is not None
        assert observability.get_current_trace_id() == custom_trace_id
        with observability.span("agent.feedback", input=custom_trace_id, trace_id=custom_trace_id) as s2:
            assert s2 is not None
            assert observability.get_current_trace_id() == custom_trace_id

    observability.flush()


# --- agent wiring: spans don't change episode behavior ------------------------


@pytest.mark.asyncio
async def test_agent_run_unaffected_by_observability_when_disabled():
    # sanity: with tracing off (the default in every other test), Agent.run's
    # `with observability.span(...)` wrapper must be fully transparent.
    import httpx
    import respx

    from agentic_rl.capabilities.http_call import HttpCallCapability
    from agentic_rl.capabilities.registry import CapabilityRegistry
    from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
    from agentic_rl.core.agent import Agent
    from agentic_rl.core.config import Mode
    from agentic_rl.core.memory import Consolidator, MemoryStore
    from agentic_rl.core.models import Candidate
    from agentic_rl.core.store import EpisodeStore
    from agentic_rl.llm.distiller import MockDistiller
    from agentic_rl.llm.mock import MockPlanner
    from agentic_rl.policy.greedy import GreedyPolicy

    class FakeScheduler:
        def add_job(self, instruction, *, cron=None, run_at=None, timezone=None):
            return "job-1"

    with respx.mock:
        respx.get("https://x").mock(return_value=httpx.Response(200, json={}))
        registry = CapabilityRegistry()
        registry.register(HttpCallCapability(httpx.AsyncClient()))
        registry.register(ScheduleTaskCapability(FakeScheduler()))
        candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
        store = EpisodeStore(":memory:")
        memory = MemoryStore(store.connection)
        agent = Agent(
            MockPlanner(default=candidates),
            GreedyPolicy(),
            registry,
            store,
            Settings(mode=Mode.DEV),
            memory,
            Consolidator(memory, MockDistiller()),
        )
        episode = await agent.run("fetch x")

    assert episode.status == "executed"
    assert episode.steps[0].outcome.ok is True
