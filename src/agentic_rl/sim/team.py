"""Multi-agent simulator scenarios (docs/MULTI-AGENT-PLAN.md, verification) — same
fake environment, mock planner and scripted users as sim/run.py, several agents.

- transfer: agent `a` trains alone, then a fresh agent `b` starts on the same
  requests. With sharing, `b` should start near `a`'s final reward.
- conflict: `a` and `b` serve users who disagree about confirming orders. Learned
  trust should drop on exactly those arms, so neither ends up worse than without
  sharing.
- routing: three specialists, requests with no agent named. The Router should learn
  to send each request to the agent that has the capability it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.hub import KnowledgeHub
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import AgentProfile, Episode, Feedback
from agentic_rl.core.router import Router
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.core.team import Team
from agentic_rl.llm.distiller import MockDistiller
from agentic_rl.llm.mock import MockPlanner
from agentic_rl.llm.summarizer import MockSummarizer
from agentic_rl.policy.linucb import LinUCBPolicy
from agentic_rl.sim import env
from agentic_rl.sim.run import REQUESTS, _default_fn, _FakeScheduler
from agentic_rl.sim.user import ScriptedUser

# which capability each sim request needs — the "right specialist" for routing
REQUEST_CAPABILITY = {
    REQUESTS[0][0]: "http_call",
    REQUESTS[1][0]: "http_call",
    REQUESTS[2][0]: "schedule_task",
}


@dataclass
class SimTeam:
    team: Team
    store: EpisodeStore
    memory: MemoryStore
    hub: KnowledgeHub


def build_team(profiles: list[AgentProfile], share: bool = True, router_prior_weight: float = 0.5) -> SimTeam:
    store = EpisodeStore(":memory:")
    registry = CapabilityRegistry()
    registry.register(HttpCallCapability(env.make_client()))
    registry.register(ScheduleTaskCapability(_FakeScheduler()))
    settings = Settings(
        mode=Mode.SIM,
        planner="mock",
        share_knowledge=share,
        router_prior_weight=router_prior_weight,
        delegation_enabled=False,
    )
    memory = MemoryStore(store.connection)
    consolidator = Consolidator(memory, MockDistiller(), default_scope="private")
    sessions = SessionStore(store.connection)
    hub = KnowledgeHub(
        store.connection, prior=settings.trust_prior, beta=settings.trust_beta, min_obs=settings.trust_min_obs, enabled=share
    )
    planner = MockPlanner(default_fn=_default_fn)
    agents = []
    for profile in profiles:
        policy = LinUCBPolicy()
        agents.append(
            Agent(
                planner, policy, registry, store, settings, memory, consolidator, sessions, MockSummarizer(),
                profile=profile, hub=hub,
            )
        )
        hub.register(profile.id, policy, share=profile.share)
    router = Router(profiles, prior_weight=router_prior_weight)
    return SimTeam(Team(agents, store, settings, hub, router), store, memory, hub)


async def _grade(team: Team, episode: Episode, user: ScriptedUser) -> Episode:
    """Scripted feedback. A write the agent asked to confirm is confirmed, then graded
    like any other (same as sim/run.py). An episode stuck at the gate because the
    agent lacks the capability it proposed is a thumbs-down."""
    if episode.status == "pending_confirmation":
        capability = episode.steps[-1].action.candidate.capability
        if team.get(episode.agent_id).registry.get_or_none(capability) is not None:
            episode = await team.confirm(episode.id)
    if episode.status == "pending_confirmation":
        score, correction = -1, None
    else:
        score, correction = user.grade(episode.steps[0].action.candidate)
    return await team.record_feedback(Feedback(episode_id=episode.id, score=score, correction=correction))


async def run_transfer(share: bool, episodes_a: int = 150, episodes_b: int = 60) -> list[float]:
    """Returns agent `b`'s per-episode reward after `a` trained alone first."""
    sim = build_team([AgentProfile(id="a"), AgentProfile(id="b")], share=share)
    user = ScriptedUser()
    for i in range(episodes_a):
        request, _ = REQUESTS[i % len(REQUESTS)]
        await _grade(sim.team, await sim.team.run(request, agent_id="a"), user)
    rewards = []
    for i in range(episodes_b):
        request, _ = REQUESTS[i % len(REQUESTS)]
        episode = await _grade(sim.team, await sim.team.run(request, agent_id="b"), user)
        rewards.append(episode.final_reward)
    return rewards


async def run_conflict(share: bool, rounds: int = 150) -> tuple[dict[str, list[float]], SimTeam]:
    """Both agents answer every request, each graded by its own user."""
    sim = build_team([AgentProfile(id="a"), AgentProfile(id="b")], share=share)
    users = {"a": ScriptedUser(order_confirmation=True), "b": ScriptedUser(order_confirmation=False)}
    rewards: dict[str, list[float]] = {"a": [], "b": []}
    for i in range(rounds):
        request, _ = REQUESTS[i % len(REQUESTS)]
        for agent_id, user in users.items():
            episode = await _grade(sim.team, await sim.team.run(request, agent_id=agent_id), user)
            rewards[agent_id].append(episode.final_reward)
    return rewards, sim


ROUTING_PROFILES = [
    AgentProfile(id="generalist", role="answers directly", capabilities=[]),
    AgentProfile(id="api", role="calls HTTP APIs", capabilities=["http_call"]),
    AgentProfile(id="scheduler", role="schedules tasks", capabilities=["schedule_task"]),
]


async def run_routing(episodes: int = 300, router_prior_weight: float = 0.0) -> list[int]:
    """1/0 per episode: did the Router pick an agent that has the capability the
    request needs? The cue prior is off by default so this measures learning alone."""
    sim = build_team(ROUTING_PROFILES, share=True, router_prior_weight=router_prior_weight)
    user = ScriptedUser()
    caps = {p.id: set(p.capabilities or []) for p in ROUTING_PROFILES}
    correct = []
    for i in range(episodes):
        request, _ = REQUESTS[i % len(REQUESTS)]
        episode = await sim.team.run(request)
        correct.append(int(REQUEST_CAPABILITY[request] in caps[episode.agent_id]))
        await _grade(sim.team, episode, user)
    return correct
