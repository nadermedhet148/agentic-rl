from __future__ import annotations

import httpx
import pytest
import respx

from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import Candidate, Feedback
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.distiller import MockDistiller
from agentic_rl.llm.mock import MockPlanner
from agentic_rl.policy.greedy import GreedyPolicy


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self._next_id = 0

    def add_job(self, instruction, *, cron=None, run_at=None, timezone=None) -> str:
        self._next_id += 1
        job_id = f"job-{self._next_id}"
        self.jobs[job_id] = {"instruction": instruction, "cron": cron, "run_at": run_at}
        return job_id


def make_agent(mode: Mode = Mode.DEV, rules=None, default=None, store: EpisodeStore | None = None):
    registry = CapabilityRegistry()
    registry.register(HttpCallCapability(httpx.AsyncClient()))
    registry.register(ScheduleTaskCapability(FakeScheduler()))
    planner = MockPlanner(rules=rules or [], default=default or [])
    policy = GreedyPolicy()
    store = store or EpisodeStore(":memory:")
    settings = Settings(mode=mode)
    memory = MemoryStore(store.connection)
    consolidator = Consolidator(memory, MockDistiller())
    return Agent(planner, policy, registry, store, settings, memory, consolidator), store


# --- read-tier auto-execute ----------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_read_tier_executes_immediately():
    respx.get("https://api.example.com/thing").mock(return_value=httpx.Response(200, json={"ok": True}))
    candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://api.example.com/thing"})]
    agent, store = make_agent(default=candidates)

    episode = await agent.run("fetch the thing")

    assert episode.status == "executed"
    assert episode.outcome.ok is True
    assert episode.implicit_reward > 0
    assert store.get(episode.id) is not None


# --- write-tier confirmation gate ----------------------------------------------


@pytest.mark.asyncio
async def test_write_tier_needs_confirmation_when_planner_flags_it():
    candidates = [
        Candidate(
            capability="http_call",
            params={"method": "POST", "url": "https://api.example.com/orders"},
            needs_confirmation=True,
        )
    ]
    agent, _store = make_agent(mode=Mode.DEV, default=candidates)

    episode = await agent.run("place an order")

    assert episode.status == "pending_confirmation"
    assert episode.outcome is None


@pytest.mark.asyncio
@respx.mock
async def test_confirm_executes_a_pending_episode():
    respx.post("https://api.example.com/orders").mock(return_value=httpx.Response(201, json={"id": 1}))
    candidates = [
        Candidate(
            capability="http_call",
            params={"method": "POST", "url": "https://api.example.com/orders"},
            needs_confirmation=True,
        )
    ]
    agent, _store = make_agent(mode=Mode.DEV, default=candidates)

    pending = await agent.run("place an order")
    assert pending.status == "pending_confirmation"

    executed = await agent.confirm(pending.id)
    assert executed.status == "executed"
    assert executed.outcome.ok is True


@pytest.mark.asyncio
async def test_confirm_unknown_episode_raises():
    agent, _ = make_agent()
    with pytest.raises(KeyError):
        await agent.confirm("nope")


@pytest.mark.asyncio
async def test_confirm_already_executed_raises():
    candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
    with respx.mock:
        respx.get("https://x").mock(return_value=httpx.Response(200, json={}))
        agent, _ = make_agent(default=candidates)
        episode = await agent.run("fetch x")

    with pytest.raises(ValueError):
        await agent.confirm(episode.id)


# --- prod_strict always confirms writes ----------------------------------------


@pytest.mark.asyncio
async def test_prod_strict_always_confirms_write_tier_even_without_planner_flag():
    candidates = [
        Candidate(
            capability="http_call",
            params={"method": "POST", "url": "https://api.example.com/orders"},
            needs_confirmation=False,  # planner didn't ask, but prod-strict overrides
        )
    ]
    agent, _ = make_agent(mode=Mode.PROD_STRICT, default=candidates)

    episode = await agent.run("place an order")

    assert episode.status == "pending_confirmation"


# --- unknown capability --------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_capability_forces_confirmation_then_fails_on_confirm():
    candidates = [Candidate(capability="delete_universe", params={})]
    agent, _ = make_agent(default=candidates)

    pending = await agent.run("do something dangerous")
    assert pending.status == "pending_confirmation"

    executed = await agent.confirm(pending.id)
    assert executed.status == "executed"
    assert executed.outcome.ok is False
    assert "unknown capability" in executed.outcome.error


# --- feedback -------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_record_feedback_sets_final_reward():
    respx.get("https://x").mock(return_value=httpx.Response(200, json={}))
    candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
    agent, store = make_agent(default=candidates)

    episode = await agent.run("fetch x")
    updated = await agent.record_feedback(Feedback(episode_id=episode.id, score=1))

    assert updated.final_reward == 1.0
    assert store.get(episode.id).final_reward == 1.0


# --- reissue penalty ------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_reissuing_same_request_penalizes_prior_episode():
    respx.get("https://x").mock(return_value=httpx.Response(200, json={}))
    candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
    agent, store = make_agent(default=candidates)

    first = await agent.run("fetch x")
    original_reward = store.get(first.id).implicit_reward

    await agent.run("fetch x")  # reissue

    penalized = store.get(first.id)
    assert penalized.implicit_reward < original_reward


# --- memory: corrections become rules, rules reach the planner ----------------


@pytest.mark.asyncio
@respx.mock
async def test_correction_produces_a_rule_visible_on_the_next_run():
    respx.get("https://x").mock(return_value=httpx.Response(200, json={}))
    candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
    agent, _store = make_agent(default=candidates)

    class RecordingPlanner(MockPlanner):
        def __init__(self):
            super().__init__(default=candidates)
            self.seen_rules: list[list[str]] = []

        async def plan(self, state, tool_schemas, prior_corrections, rules=None):
            self.seen_rules.append(list(rules or []))
            return await super().plan(state, tool_schemas, prior_corrections, rules)

    recording_planner = RecordingPlanner()
    agent._planner = recording_planner  # swap in a spy without rebuilding the whole fixture

    episode = await agent.run("fetch x")
    assert recording_planner.seen_rules[0] == []  # no rules exist yet

    await agent.record_feedback(
        Feedback(episode_id=episode.id, score=-1, correction="always send Accept: application/json")
    )

    await agent.run("fetch x again")
    assert "always send Accept: application/json" in recording_planner.seen_rules[-1]


@pytest.mark.asyncio
async def test_feedback_without_correction_does_not_create_a_rule():
    candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
    with respx.mock:
        respx.get("https://x").mock(return_value=httpx.Response(200, json={}))
        agent, _store = make_agent(default=candidates)
        episode = await agent.run("fetch x")

    await agent.record_feedback(Feedback(episode_id=episode.id, score=1, correction=None))
    assert agent._memory.active_rules() == []


# --- scheduled task cancellation -----------------------------------------------


@pytest.mark.asyncio
async def test_cancel_task_penalizes_scheduling_episode():
    candidates = [
        Candidate(capability="schedule_task", params={"instruction": "ping", "cron": "0 9 * * *"})
    ]
    agent, _store = make_agent(default=candidates)

    episode = await agent.run("ping me every morning")
    assert episode.status == "executed"
    job_id = episode.outcome.payload["job_id"]
    original_reward = episode.implicit_reward

    penalized = agent.cancel_task(job_id)

    assert penalized is not None
    assert penalized.implicit_reward < original_reward
