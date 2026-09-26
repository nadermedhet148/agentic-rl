from __future__ import annotations

import json
import statistics

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient

from agentic_rl.api.app import create_app
from agentic_rl.capabilities.delegate import DelegateCapability
from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.hub import KnowledgeHub
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import AgentProfile, Candidate, Episode, Feedback, State, Step
from agentic_rl.core.router import Router
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.core.team import Team
from agentic_rl.llm.distiller import MockDistiller, loosens_safety_heuristic
from agentic_rl.llm.mock import MockPlanner
from agentic_rl.llm.summarizer import MockSummarizer
from agentic_rl.policy import features
from agentic_rl.policy.base import Arm, PeerEvidence
from agentic_rl.policy.epsilon import EpsilonGreedyPolicy
from agentic_rl.policy.linucb import LinUCBPolicy
from agentic_rl.sim import team as sim_team

# --- helpers ---------------------------------------------------------------------------


def _arm(arm_id: str, dim: int = 4) -> Arm:
    x = np.zeros(dim)
    x[0] = 1.0
    return Arm(id=arm_id, features=x, confidence=0.5)


def make_team(
    profiles: list[AgentProfile],
    default_fn,
    mode: Mode = Mode.DEV,
    delegation: bool = True,
    share: bool = True,
    store: EpisodeStore | None = None,
):
    store = store or EpisodeStore(":memory:")
    registry = CapabilityRegistry()
    registry.register(HttpCallCapability(httpx.AsyncClient()))
    settings = Settings(mode=mode, planner="mock", share_knowledge=share)
    memory = MemoryStore(store.connection)
    consolidator = Consolidator(memory, MockDistiller(), default_scope="private")
    sessions = SessionStore(store.connection)
    hub = KnowledgeHub(store.connection, enabled=share)
    planner = MockPlanner(default_fn=default_fn)
    agents, delegates = [], []
    for profile in profiles:
        policy = LinUCBPolicy()
        extras = []
        if delegation:
            delegate = DelegateCapability(profile.id, profiles, max_depth=settings.max_delegation_depth)
            delegates.append(delegate)
            extras.append(delegate)
        agents.append(
            Agent(
                planner, policy, registry, store, settings, memory, consolidator, sessions, MockSummarizer(),
                profile=profile, hub=hub, extra_capabilities=extras,
            )
        )
        hub.register(profile.id, policy, share=profile.share)
    team = Team(agents, store, settings, hub, Router(profiles))
    for delegate in delegates:
        delegate.bind(team.delegate)
    return team, store, memory, hub


def _answer(text: str = "done") -> Candidate:
    return Candidate(capability="answer", params={"text": text}, confidence=0.9)


# --- policy evidence pooling -------------------------------------------------------------


def test_linucb_evidence_excludes_prior_and_pooled_peer_evidence():
    a = LinUCBPolicy(dim=4)
    a.update(_arm("x"), 1.0)
    ev = a.evidence()
    assert ev["kind"] == "linucb"
    assert np.allclose(np.asarray(ev["arms"]["x"]["A"]), np.outer(_arm("x").features, _arm("x").features))

    b = LinUCBPolicy(dim=4)
    b.set_peer_evidence([PeerEvidence("a", ev, lambda _arm_id: 1.0)])
    assert b.evidence()["arms"] == {}  # pooled evidence is never re-shared
    assert b.predict(_arm("x")) is None  # nor counted as b's own knowledge


def test_linucb_pooling_is_idempotent_and_equals_seeing_the_data():
    a = LinUCBPolicy(dim=4)
    for _ in range(5):
        a.update(_arm("good"), 1.0)
        a.update(_arm("bad"), -1.0)
    peers = [PeerEvidence("a", a.evidence(), lambda _arm_id: 1.0)]

    b = LinUCBPolicy(dim=4)
    b.set_peer_evidence(peers)
    first = b.scores([_arm("good"), _arm("bad")])
    b.set_peer_evidence(peers)
    b.set_peer_evidence(peers)
    assert np.allclose(b.scores([_arm("good"), _arm("bad")]), first)  # no double counting
    assert np.allclose(first, a.scores([_arm("good"), _arm("bad")]))  # full trust == a's own data
    assert b.select([_arm("bad"), _arm("good")])[0] == 1


def test_linucb_ignores_other_policy_kinds_and_zero_weight():
    eps = EpsilonGreedyPolicy()
    eps.update(_arm("x"), 1.0)
    lin = LinUCBPolicy(dim=4)
    lin.set_peer_evidence([PeerEvidence("e", eps.evidence(), lambda _arm_id: 1.0)])
    other = LinUCBPolicy(dim=4)
    other.update(_arm("x"), 5.0)
    lin.set_peer_evidence([PeerEvidence("o", other.evidence(), lambda _arm_id: 0.0)])
    assert np.allclose(lin.scores([_arm("x")])[0], [0.0])


def test_epsilon_pooling_uses_trust_weighted_counts():
    a = EpsilonGreedyPolicy(epsilon=0.0)
    for _ in range(4):
        a.update(_arm("x"), 1.0)
    b = EpsilonGreedyPolicy(epsilon=0.0)
    b.update(_arm("x"), -1.0)
    b.set_peer_evidence([PeerEvidence("a", a.evidence(), lambda _arm_id: 0.5)])
    # (1 * -1 + 0.5 * 4 * 1) / (1 + 2) = 1/3
    assert b._pooled_mean("x") == pytest.approx(1 / 3)
    assert b.predict(_arm("x")) == pytest.approx(-1.0)


# --- trust -----------------------------------------------------------------------------------


def test_trust_starts_at_prior_and_drops_for_a_misleading_peer(tmp_path):
    store = EpisodeStore(tmp_path / "t.db")
    hub = KnowledgeHub(store.connection, prior=0.5, min_obs=2)
    a, b = LinUCBPolicy(dim=4), LinUCBPolicy(dim=4)
    hub.register("a", a)
    hub.register("b", b)
    for _ in range(10):
        a.update(_arm("x"), 3.0)  # a's user loves x
    assert hub.trust("b", "a", "x") == 0.5

    for _ in range(10):  # b's user hates x
        hub.observe("b", _arm("x"), -3.0)
        b.update(_arm("x"), -3.0)
    assert hub.trust("b", "a", "x") < 0.1
    assert hub.trust("b", "a", "unseen-arm") < 0.1  # falls back to the pair-level estimate

    reloaded = KnowledgeHub(store.connection, prior=0.5, min_obs=2)  # persisted
    reloaded.register("a", a)
    reloaded.register("b", b)
    assert reloaded.trust("b", "a", "x") == pytest.approx(hub.trust("b", "a", "x"))


def test_trust_stays_high_for_an_agreeing_peer():
    store = EpisodeStore(":memory:")
    hub = KnowledgeHub(store.connection, min_obs=2)
    a, b = LinUCBPolicy(dim=4), LinUCBPolicy(dim=4)
    hub.register("a", a)
    hub.register("b", b)
    for _ in range(10):
        a.update(_arm("x"), 1.0)
    for _ in range(5):
        hub.observe("b", _arm("x"), 1.0)
        b.update(_arm("x"), 1.0)
    assert hub.trust("b", "a", "x") == 1.0


def test_non_sharing_agent_neither_gives_nor_takes():
    store = EpisodeStore(":memory:")
    hub = KnowledgeHub(store.connection)
    a, b, c = LinUCBPolicy(dim=4), LinUCBPolicy(dim=4), LinUCBPolicy(dim=4)
    hub.register("a", a)
    hub.register("b", b)
    hub.register("loner", c, share=False)
    assert set(hub.peer_weights("a")) == {"b"}
    assert hub.peer_weights("loner") == {}
    hub.enabled = False
    assert hub.peer_weights("a") == {}


# --- simulator: do they actually learn from each other? ---------------------------------------


@pytest.mark.asyncio
async def test_sim_transfer_fresh_agent_starts_where_its_peer_left_off():
    shared = await sim_team.run_transfer(True, episodes_a=90, episodes_b=15)
    isolated = await sim_team.run_transfer(False, episodes_a=90, episodes_b=15)
    assert statistics.mean(shared) > statistics.mean(isolated)
    assert statistics.mean(shared) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_sim_conflict_no_negative_transfer_and_trust_drops_where_users_disagree():
    shared, sim = await sim_team.run_conflict(True, rounds=90)
    isolated, _ = await sim_team.run_conflict(False, rounds=90)
    for agent_id in ("a", "b"):
        assert statistics.mean(shared[agent_id][-30:]) >= statistics.mean(isolated[agent_id][-30:])
    assert statistics.mean(shared["a"] + shared["b"]) >= statistics.mean(isolated["a"] + isolated["b"]) - 0.05

    order = {"method": "POST", "url": "https://api.example.com/orders", "json_body": {"item": "widget"}}
    confirm_arm = features.arm_id(Candidate(capability="http_call", params=order, needs_confirmation=True))
    data = {"method": "GET", "url": "https://api.example.com/api/data", "headers": {"Accept": "application/json"}}
    data_arm = features.arm_id(Candidate(capability="http_call", params=data))
    assert sim.hub.trust("b", "a", confirm_arm) < 0.2
    assert sim.hub.trust("b", "a", data_arm) > 0.8


@pytest.mark.asyncio
async def test_sim_router_learns_which_specialist_to_pick():
    correct = await sim_team.run_routing(episodes=150)
    assert statistics.mean(correct[-60:]) > 0.9
    assert statistics.mean(correct[-60:]) >= statistics.mean(correct[:15])


# --- router -------------------------------------------------------------------------------


def test_router_capability_prior_routes_sensibly_before_any_feedback():
    profiles = [
        AgentProfile(id="researcher", capabilities=["web_search"]),
        AgentProfile(id="integrator", capabilities=["http_call", "schedule_task"]),
    ]
    router = Router(profiles, prior_weight=0.5)
    assert router.select("fetch https://api.example.com/x", explore=False)[0] == "integrator"
    assert router.select("search the latest news about rust", explore=False)[0] == "researcher"


def test_router_learning_overrides_the_prior():
    profiles = [AgentProfile(id="a", capabilities=["http_call"]), AgentProfile(id="b", capabilities=[])]
    router = Router(profiles, prior_weight=0.5)
    request = "fetch https://api.example.com/x"
    for _ in range(20):
        router.update(request, "user", "a", -3.0)
        router.update(request, "user", "b", 3.0)
    assert router.select(request, explore=False)[0] == "b"
    restored = Router(profiles, prior_weight=0.5)
    restored.load_state(router.state_dict())
    assert restored.select(request, explore=False)[0] == "b"


# --- semantic sharing: rule promotion ------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_correction_from_two_agents_becomes_one_team_rule():
    team, _store, memory, _ = make_team(
        [AgentProfile(id="a"), AgentProfile(id="b")], lambda s, h: [_answer()], delegation=False
    )
    for agent_id in ("a", "b"):
        episode = await team.run("say hello", agent_id=agent_id)
        await team.record_feedback(Feedback(episode_id=episode.id, score=-1, correction="always greet in French"))

    (rule,) = memory.active_rules()
    assert rule.scope == "team"
    assert rule.support_by_agent == {"a": 1, "b": 1}


@pytest.mark.asyncio
async def test_rule_is_private_until_a_peer_agrees():
    team, _store, memory, _ = make_team(
        [AgentProfile(id="a"), AgentProfile(id="b")], lambda s, h: [_answer()], delegation=False
    )
    episode = await team.run("say hello", agent_id="a")
    await team.record_feedback(Feedback(episode_id=episode.id, score=-1, correction="always greet in French"))
    (rule,) = memory.active_rules()
    assert rule.scope == "private"
    assert memory.active_rules(agent_id="b") == []


@pytest.mark.asyncio
async def test_safety_loosening_rule_is_never_auto_promoted_but_a_human_can():
    team, _store, memory, _ = make_team(
        [AgentProfile(id="a"), AgentProfile(id="b")], lambda s, h: [_answer()], delegation=False
    )
    correction = "don't ask me to confirm before placing orders"
    for agent_id in ("a", "b"):
        episode = await team.run("order a widget", agent_id=agent_id)
        await team.record_feedback(Feedback(episode_id=episode.id, score=-1, correction=correction))
    (rule,) = memory.active_rules()
    assert rule.loosens_safety
    assert rule.scope == "private"
    assert rule.support_by_agent == {"a": 1, "b": 1}

    assert memory.promote(rule.id).scope == "team"


def test_loosens_safety_heuristic():
    assert loosens_safety_heuristic("don't ask before POSTing")
    assert loosens_safety_heuristic("no need to confirm orders")
    assert loosens_safety_heuristic("skip the confirmation step")
    assert not loosens_safety_heuristic("always require confirmation before POSTing")
    assert not loosens_safety_heuristic("always send Accept: application/json")


def test_private_override_hides_a_team_rule_for_its_owner_only():
    store = EpisodeStore(":memory:")
    memory = MemoryStore(store.connection)
    team_rule = memory.add("use metric units")
    override = memory.add("use imperial units", owner_agent_id="a", scope="private", overrides_id=team_rule.id)

    assert {m.id for m in memory.active_rules(agent_id="a")} == {override.id}
    assert {m.id for m in memory.active_rules(agent_id="b")} == {team_rule.id}

    memory.promote(override.id)
    assert memory.get(team_rule.id).active is False  # promotion replaces it for everyone
    assert {m.id for m in memory.active_rules(agent_id="b")} == {override.id}


def test_capability_scoped_rules_only_reach_agents_with_that_capability():
    store = EpisodeStore(":memory:")
    memory = MemoryStore(store.connection)
    general = memory.add("be concise")
    http = memory.add("send Accept: application/json", capability="http_call")
    assert {m.id for m in memory.active_rules(capabilities=["web_search", "answer"])} == {general.id}
    assert {m.id for m in memory.active_rules(capabilities=["http_call"])} == {general.id, http.id}


# --- episodic sharing -----------------------------------------------------------------------------


def _saved_episode(store, agent_id, request, correction=None, reward=None, steps=None):
    episode = Episode(state=State(request=request), agent_id=agent_id, steps=steps or [])
    episode.correction = correction
    episode.final_reward = reward
    if reward is not None:
        episode.explicit_score = 1 if reward > 0 else -1  # user-graded, like a real demonstration
    store.save(episode)
    return episode


def test_corrections_own_first_then_trusted_peers_labelled():
    store = EpisodeStore(":memory:")
    _saved_episode(store, "a", "fetch weather berlin", correction="use celsius")
    _saved_episode(store, "b", "fetch weather berlin today", correction="include humidity")
    _saved_episode(store, "c", "fetch weather berlin now", correction="untrusted advice")

    results = store.search_corrections(
        "weather berlin", agent_id="a", peer_weights={"b": 0.9, "c": 0.1}, min_trust=0.2
    )
    assert results[0].endswith("use celsius")
    assert results[1].startswith("(from peer agent 'b')")
    assert not any("untrusted" in r for r in results)
    assert len(store.search_corrections("weather berlin")) == 3  # no agent: unfiltered, as before


def _step(capability: str, params: dict) -> Step:
    from agentic_rl.core.models import Action, Outcome

    candidate = Candidate(capability=capability, params=params)
    return Step(
        index=0,
        candidates=[candidate],
        action=Action(candidate=candidate, index=0, explored=False, arm_id="x"),
        outcome=Outcome(ok=True),
    )


def test_demonstrations_come_from_trusted_peers_with_usable_capabilities():
    store = EpisodeStore(":memory:")
    _saved_episode(store, "b", "weather in berlin", reward=1.0, steps=[_step("web_search", {"query": "berlin"})])
    _saved_episode(store, "b", "weather in paris", reward=1.0, steps=[_step("http_call", {"url": "u"})])
    _saved_episode(store, "b", "weather in rome", reward=-1.0, steps=[_step("web_search", {"query": "rome"})])
    _saved_episode(store, "a", "weather in oslo", reward=1.0, steps=[_step("web_search", {"query": "oslo"})])

    demos = store.search_demonstrations(
        "weather", "a", {"b": 0.9}, capabilities={"web_search", "answer"}, min_trust=0.2
    )
    assert len(demos) == 1
    assert "berlin" in demos[0] and "web_search" in demos[0]
    assert store.search_demonstrations("weather", "a", {"b": 0.1}, min_trust=0.2) == []


@pytest.mark.asyncio
async def test_planner_gets_persona_and_peer_demonstrations():
    seen: dict = {}

    class Spy(MockPlanner):
        async def plan(self, state, tool_schemas, prior_corrections, rules=None, history=None,
                       conversation_summary="", conversation_turns=None, persona="", demonstrations=None):
            seen.setdefault("persona", persona)
            seen.setdefault("demonstrations", demonstrations)
            return [_answer()]

    team, store, _memory, _ = make_team(
        [AgentProfile(id="a", persona="You research things."), AgentProfile(id="b")],
        lambda s, h: [_answer()],
        delegation=False,
    )
    _saved_episode(store, "b", "weather in berlin", reward=1.0, steps=[_step("http_call", {"url": "u"})])
    team.get("a")._planner = Spy()

    await team.run("weather in berlin", agent_id="a")
    assert seen["persona"] == "You research things."
    assert seen["demonstrations"] and "berlin" in seen["demonstrations"][0]


# --- delegation --------------------------------------------------------------------------------------

BOSS = AgentProfile(id="boss", role="coordinator", capabilities=[])
WORKER = AgentProfile(id="worker", role="calls APIs", capabilities=["http_call"])


def _delegating_planner(worker_candidate: Candidate):
    def fn(state: State, history: list[Step]) -> list[Candidate]:
        if state.request.startswith("sub:"):
            return [_answer("worker result")] if history else [worker_candidate]
        if history:
            return [_answer(f"boss says: {history[-1].outcome.payload.get('answer')}")]
        return [Candidate(capability="delegate", params={"agent_id": "worker", "request": "sub: do it"})]

    return fn


@pytest.mark.asyncio
@respx.mock
async def test_delegation_runs_peer_and_uses_its_answer():
    respx.get("https://api.example.com/data").mock(return_value=httpx.Response(200, json={"ok": 1}))
    get = Candidate(capability="http_call", params={"method": "GET", "url": "https://api.example.com/data"})
    team, store, _memory, _ = make_team([BOSS, WORKER], _delegating_planner(get))

    parent = await team.run("do the thing", agent_id="boss")

    assert parent.status == "executed"
    assert parent.answer == "boss says: worker result"
    child_id = parent.steps[0].child_episode_id
    child = store.get(child_id)
    assert child.agent_id == "worker"
    assert child.parent_episode_id == parent.id
    assert child.delegation_depth == 1
    assert child.session_id is None


@pytest.mark.asyncio
@respx.mock
async def test_delegated_write_pauses_the_parent_and_confirming_parent_resumes_both():
    route = respx.post("https://api.example.com/orders").mock(return_value=httpx.Response(201, json={"id": 1}))
    post = Candidate(
        capability="http_call",
        params={"method": "POST", "url": "https://api.example.com/orders"},
        needs_confirmation=True,
    )
    team, store, _memory, _ = make_team([BOSS, WORKER], _delegating_planner(post))

    parent = await team.run("place the order", agent_id="boss")

    assert parent.status == "pending_confirmation"  # the write was not laundered through delegation
    assert parent.steps[0].outcome is None
    child = store.get(parent.steps[0].child_episode_id)
    assert child.status == "pending_confirmation"
    assert not route.called

    resumed = await team.confirm(parent.id)

    assert route.called
    assert store.get(child.id).status == "executed"
    assert resumed.status == "executed"
    assert resumed.answer == "boss says: worker result"


@pytest.mark.asyncio
@respx.mock
async def test_confirming_the_child_directly_also_resumes_the_parent():
    respx.post("https://api.example.com/orders").mock(return_value=httpx.Response(201, json={"id": 1}))
    post = Candidate(
        capability="http_call",
        params={"method": "POST", "url": "https://api.example.com/orders"},
        needs_confirmation=True,
    )
    team, store, _memory, _ = make_team([BOSS, WORKER], _delegating_planner(post))
    parent = await team.run("place the order", agent_id="boss")

    await team.confirm(parent.steps[0].child_episode_id)

    assert store.get(parent.id).status == "executed"


@pytest.mark.asyncio
async def test_delegation_depth_limit_and_unknown_peer():
    def fn(state, history):
        if history:
            return [_answer(str(history[-1].outcome.error))]
        if state.request.startswith("sub:"):
            return [Candidate(capability="delegate", params={"agent_id": "boss", "request": "sub: again"})]
        return [Candidate(capability="delegate", params={"agent_id": "worker", "request": "sub: go"})]

    team, store, _memory, _ = make_team([BOSS, WORKER], fn)
    parent = await team.run("start", agent_id="boss")
    child = store.get(parent.steps[0].child_episode_id)
    assert "depth limit" in child.answer

    unknown = DelegateCapability("boss", [BOSS, WORKER])
    unknown.bind(team.delegate)
    assert (await unknown.execute({"agent_id": "nobody", "request": "x"})).ok is False
    assert "worker" in unknown.input_schema["properties"]["agent_id"]["enum"]
    assert "boss" not in unknown.input_schema["properties"]["agent_id"]["enum"]


@pytest.mark.asyncio
@respx.mock
async def test_feedback_on_parent_credits_the_child_at_a_discount():
    respx.get("https://api.example.com/data").mock(return_value=httpx.Response(200, json={"ok": 1}))
    get = Candidate(capability="http_call", params={"method": "GET", "url": "https://api.example.com/data"})
    team, store, _memory, _ = make_team([BOSS, WORKER], _delegating_planner(get))
    parent = await team.run("do the thing", agent_id="boss")

    await team.record_feedback(Feedback(episode_id=parent.id, score=1))

    child = store.get(parent.steps[0].child_episode_id)
    assert child.final_reward == pytest.approx(3.0 * 0.5)  # weighted parent reward * delegation_credit


def test_delegate_arm_identity_is_per_peer():
    to_a = Candidate(capability="delegate", params={"agent_id": "a", "request": "x"})
    to_a_other = Candidate(capability="delegate", params={"agent_id": "a", "request": "y"})
    to_b = Candidate(capability="delegate", params={"agent_id": "b", "request": "x"})
    assert features.arm_id(to_a) == features.arm_id(to_a_other)
    assert features.arm_id(to_a) != features.arm_id(to_b)


# --- API ----------------------------------------------------------------------------------------------


def _write_agents(tmp_path, profiles):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps(profiles))
    return path


@respx.mock
def test_api_team_endpoints(tmp_path):
    respx.get("https://example.com/thing").mock(return_value=httpx.Response(200, json={"ok": True}))
    agents_file = _write_agents(
        tmp_path,
        [
            {"id": "researcher", "role": "researches", "capabilities": ["web_search"]},
            {"id": "integrator", "role": "calls APIs", "capabilities": ["http_call", "schedule_task"]},
        ],
    )
    settings = Settings(mode=Mode.DEV, db_path=tmp_path / "db.sqlite", planner="mock", agents_file=agents_file)
    with TestClient(create_app(settings)) as client:
        agents = client.get("/agents").json()
        assert [a["id"] for a in agents] == ["researcher", "integrator"]
        assert "delegate" in agents[0]["available_capabilities"]
        assert "http_call" not in agents[0]["available_capabilities"]

        routed = client.post("/chat", json={"message": "http_call GET https://example.com/thing"}).json()
        assert routed["agent_id"] == "integrator"
        assert routed["routed"] is True

        explicit = client.post("/chat", json={"message": "hello", "agent_id": "researcher"}).json()
        assert explicit["agent_id"] == "researcher"
        assert explicit["routed"] is False
        assert client.post("/chat", json={"message": "x", "agent_id": "nobody"}).status_code == 404

        client.post("/feedback", json={"episode_id": routed["id"], "score": 1})
        assert [e["id"] for e in client.get("/episodes", params={"agent_id": "integrator"}).json()] == [routed["id"]]
        metrics = client.get("/agents/integrator/metrics").json()
        assert metrics["n"] == 1
        assert client.get("/agents/nobody/metrics").status_code == 404
        trust = client.get("/agents/trust").json()
        assert {(t["agent_id"], t["peer_id"]) for t in trust} == {
            ("researcher", "integrator"),
            ("integrator", "researcher"),
        }

        rule = client.post("/memories", json={"text": "be brief"}).json()
        assert client.post(f"/memories/{rule['id']}/demote").json()["scope"] == "private"
        assert client.post(f"/memories/{rule['id']}/promote").json()["scope"] == "team"
        assert client.post("/memories/nope/promote").status_code == 404


def test_api_rejects_duplicate_agent_ids(tmp_path):
    agents_file = _write_agents(tmp_path, [{"id": "a"}, {"id": "a"}])
    with pytest.raises(ValueError, match="duplicate"):
        create_app(Settings(db_path=tmp_path / "db.sqlite", planner="mock", agents_file=agents_file))


def test_api_without_agents_file_is_the_single_default_agent(tmp_path):
    with TestClient(create_app(Settings(db_path=tmp_path / "db.sqlite", planner="mock"))) as client:
        agents = client.get("/agents").json()
        assert [a["id"] for a in agents] == ["default"]
        assert "delegate" not in agents[0]["available_capabilities"]
        assert client.get("/agents/trust").json() == []
