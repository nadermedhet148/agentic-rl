from __future__ import annotations

import sqlite3

import httpx
import pytest

from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import DEFAULT_AGENT_ID, AgentProfile, Candidate, Feedback
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.distiller import MockDistiller
from agentic_rl.llm.mock import MockPlanner
from agentic_rl.llm.summarizer import MockSummarizer
from agentic_rl.policy.linucb import LinUCBPolicy


class FakeScheduler:
    def add_job(self, instruction, *, cron=None, run_at=None, timezone=None) -> str:
        return "job-1"


class RecordingPlanner(MockPlanner):
    """MockPlanner that also records the tool schemas it was offered."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.offered: list[str] = []

    async def plan(self, state, tool_schemas, *args, **kwargs):
        self.offered = [t["name"] for t in tool_schemas]
        return await super().plan(state, tool_schemas, *args, **kwargs)


def make_team_agent(store: EpisodeStore, profile: AgentProfile | None, default=None, registry=None):
    if registry is None:
        registry = CapabilityRegistry()
        registry.register(HttpCallCapability(httpx.AsyncClient()))
        registry.register(ScheduleTaskCapability(FakeScheduler()))
    planner = RecordingPlanner(default=default or [])
    memory = MemoryStore(store.connection)
    agent = Agent(
        planner,
        LinUCBPolicy(),
        registry,
        store,
        Settings(mode=Mode.DEV),
        memory,
        Consolidator(memory, MockDistiller()),
        SessionStore(store.connection),
        MockSummarizer(),
        profile=profile,
    )
    return agent, planner, memory


# --- registry views ------------------------------------------------------------


def test_registry_view_shares_instances_and_limits_names():
    registry = CapabilityRegistry()
    http = HttpCallCapability(httpx.AsyncClient())
    registry.register(http)
    registry.register(ScheduleTaskCapability(FakeScheduler()))

    view = registry.view(["http_call"])

    assert view.names() == ["http_call"]
    assert view.get("http_call") is http
    assert registry.names() == ["http_call", "schedule_task"]  # base untouched


def test_registry_view_rejects_unknown_capability():
    with pytest.raises(KeyError):
        CapabilityRegistry().view(["nope"])


# --- agent profile ---------------------------------------------------------------


def test_default_profile_is_the_single_agent():
    agent, _, _ = make_team_agent(EpisodeStore(":memory:"), profile=None)
    assert agent.id == DEFAULT_AGENT_ID
    assert agent.profile.capabilities is None


@pytest.mark.asyncio
async def test_profile_limits_capabilities_offered_to_planner():
    store = EpisodeStore(":memory:")
    profile = AgentProfile(id="scheduler-bot", capabilities=["schedule_task"])
    agent, planner, _ = make_team_agent(store, profile)

    await agent.run("anything")

    assert sorted(planner.offered) == ["answer", "schedule_task"]


@pytest.mark.asyncio
async def test_capability_outside_profile_is_never_auto_executed():
    store = EpisodeStore(":memory:")
    profile = AgentProfile(id="scheduler-bot", capabilities=["schedule_task"])
    get = Candidate(capability="http_call", params={"method": "GET", "url": "https://example.com"})
    agent, _, _ = make_team_agent(store, profile, default=[get])

    episode = await agent.run("fetch it")

    # a read-tier http_call would auto-execute for an agent that has it; for this
    # agent it's unknown, so it hits the confirm gate instead
    assert episode.status == "pending_confirmation"


@pytest.mark.asyncio
async def test_episode_and_policy_state_are_tagged_with_agent_id():
    store = EpisodeStore(":memory:")
    answer = Candidate(capability="answer", params={"text": "hi"})
    agent, _, _ = make_team_agent(store, AgentProfile(id="a1"), default=[answer])

    episode = await agent.run("anything")

    assert episode.status == "executed"
    assert episode.agent_id == "a1"
    assert store.get(episode.id).agent_id == "a1"
    assert [e.id for e in store.list_episodes(agent_id="a1")] == [episode.id]
    assert store.list_episodes(agent_id="someone-else") == []
    assert store.load_policy_state("linucb", agent_id="a1") is not None
    assert store.load_policy_state("linucb") is None  # nothing written for the default agent


@pytest.mark.asyncio
async def test_correction_rule_is_owned_by_the_agent_that_got_it():
    store = EpisodeStore(":memory:")
    agent, _, memory = make_team_agent(store, AgentProfile(id="a1"))
    episode = await agent.run("anything")

    await agent.record_feedback(Feedback(episode_id=episode.id, score=-1, correction="always say please"))

    (rule,) = memory.active_rules()
    assert rule.owner_agent_id == "a1"


# --- rule visibility -------------------------------------------------------------


def test_private_rules_are_visible_to_their_owner_only():
    store = EpisodeStore(":memory:")
    memory = MemoryStore(store.connection)
    team = memory.add("team rule")
    mine = memory.add("a1's rule", owner_agent_id="a1", scope="private")
    memory.add("a2's rule", owner_agent_id="a2", scope="private")

    assert {m.id for m in memory.active_rules(agent_id="a1")} == {team.id, mine.id}
    assert len(memory.active_rules()) == 3  # no agent filter = everything
    assert memory.get(mine.id).scope == "private"


# --- migrations from a pre-multi-agent database ----------------------------------


def test_pre_multi_agent_database_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE policy_state (policy_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL);
        INSERT INTO policy_state VALUES ('linucb', '{"v": 1}', '2026-01-01T00:00:00');
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, kind TEXT NOT NULL,
            text TEXT NOT NULL, capability TEXT, support_count INTEGER NOT NULL,
            source_episode_ids TEXT NOT NULL, superseded_by TEXT, active INTEGER NOT NULL
        );
        INSERT INTO memories VALUES ('m1', '2026-01-01T00:00:00', '2026-01-01T00:00:00', 'rule',
            'old rule', NULL, 1, '[]', NULL, 1);
        """
    )
    conn.commit()
    conn.close()

    store = EpisodeStore(db)
    memory = MemoryStore(store.connection)

    assert store.load_policy_state("linucb") == {"v": 1}
    assert store.load_policy_state("linucb", agent_id=DEFAULT_AGENT_ID) == {"v": 1}
    (rule,) = memory.active_rules(agent_id="any-agent")  # old rules stay visible to everyone
    assert rule.owner_agent_id == DEFAULT_AGENT_ID
    assert rule.scope == "team"
    store.close()
