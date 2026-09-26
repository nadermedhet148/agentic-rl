from __future__ import annotations

import argparse
import asyncio
import statistics

import numpy as np

from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import Candidate, Feedback, State, Step
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.distiller import MockDistiller
from agentic_rl.llm.mock import MockPlanner
from agentic_rl.llm.summarizer import MockSummarizer
from agentic_rl.policy.base import Policy
from agentic_rl.policy.epsilon import EpsilonGreedyPolicy
from agentic_rl.policy.greedy import GreedyPolicy
from agentic_rl.policy.linucb import LinUCBPolicy
from agentic_rl.sim import env
from agentic_rl.sim.user import ScriptedUser


class _FakeScheduler:
    """Just enough of SchedulerPort for schedule_task to succeed — the simulator
    grades the *scheduling call itself*, not whether jobs later fire."""

    def __init__(self) -> None:
        self._next_id = 0

    def add_job(self, instruction, *, cron=None, run_at=None, timezone=None) -> str:
        self._next_id += 1
        return f"sim-job-{self._next_id}"


def _http_data_candidates() -> list[Candidate]:
    url = "https://api.example.com/api/data"
    bad = Candidate(capability="http_call", params={"method": "GET", "url": url}, confidence=0.5)
    good = Candidate(
        capability="http_call",
        params={"method": "GET", "url": url, "headers": {"Accept": "application/json"}},
        confidence=0.5,
    )
    return [bad, good]


def _http_order_candidates() -> list[Candidate]:
    params = {"method": "POST", "url": "https://api.example.com/orders", "json_body": {"item": "widget"}}
    bad = Candidate(capability="http_call", params=params, confidence=0.5, needs_confirmation=False)
    good = Candidate(capability="http_call", params=params, confidence=0.5, needs_confirmation=True)
    return [bad, good]


def _schedule_candidates() -> list[Candidate]:
    params = {"instruction": "run health check", "cron": "*/5 * * * *"}
    bad = Candidate(capability="schedule_task", params=params, confidence=0.5)
    good = Candidate(capability="schedule_task", params={**params, "timezone": "Europe/Berlin"}, confidence=0.5)
    return [bad, good]


# (request text, candidate-pair builder) — order is [bad, good] on purpose: a
# non-learning policy that just takes index 0 (or ties on confidence) will keep
# picking the bad one, giving a clean flat baseline to compare a learning policy against.
REQUESTS: list[tuple[str, callable]] = [
    ("fetch data from https://api.example.com/api/data", _http_data_candidates),
    ("place an order at https://api.example.com/orders", _http_order_candidates),
    ("schedule the health check", _schedule_candidates),
]


def _default_fn(state: State, history: list[Step]) -> list[Candidate]:
    if history:
        return [Candidate(capability="answer", params={"text": "done"}, confidence=0.9)]
    for _request, builder in REQUESTS:
        if _request == state.request:
            return builder()
    return []


def _build_policy(policy_id: str, rng: np.random.Generator) -> Policy:
    if policy_id == "linucb":
        return LinUCBPolicy()
    if policy_id == "epsilon":
        return EpsilonGreedyPolicy(rng=rng)
    if policy_id == "greedy":
        return GreedyPolicy()
    raise ValueError(f"unknown policy: {policy_id}")


async def run_simulation(policy_id: str, episodes: int, seed: int = 0) -> tuple[list[float], MemoryStore]:
    """Runs `episodes` rounds of plan -> select -> execute -> grade -> learn against
    the fake environment (sim/env.py) and scripted user (sim/user.py), returning the
    per-episode final reward in order, plus the MemoryStore accumulated along the
    way (see docs/MEMORY-PLAN.md verification: this should settle at one rule per
    scripted preference, not one per episode)."""
    np_rng = np.random.default_rng(seed)

    store = EpisodeStore(":memory:")
    registry = CapabilityRegistry()
    registry.register(HttpCallCapability(env.make_client()))
    registry.register(ScheduleTaskCapability(_FakeScheduler()))
    planner = MockPlanner(default_fn=_default_fn)
    policy = _build_policy(policy_id, np_rng)
    settings = Settings(mode=Mode.SIM, planner="mock", policy=policy_id)
    memory = MemoryStore(store.connection)
    consolidator = Consolidator(memory, MockDistiller())
    sessions = SessionStore(store.connection)
    agent = Agent(planner, policy, registry, store, settings, memory, consolidator, sessions, MockSummarizer())
    user = ScriptedUser()

    rewards: list[float] = []
    for i in range(episodes):
        request, _builder = REQUESTS[i % len(REQUESTS)]
        episode = await agent.run(request, source="user")
        if episode.status == "pending_confirmation":
            episode = await agent.confirm(episode.id)
        score, correction = user.grade(episode.steps[0].action.candidate)
        updated = await agent.record_feedback(Feedback(episode_id=episode.id, score=score, correction=correction))
        rewards.append(updated.final_reward)

    # note: store.close() is intentionally not called here — memory is returned to
    # the caller (e.g. main() below) which still needs to query it.
    return rewards, memory


def _summarize(rewards: list[float], window: int = 100) -> None:
    n = len(rewards)
    first = rewards[: min(window, n)]
    last = rewards[max(0, n - window) :]
    print(f"episodes={n}")
    print(f"  first {len(first)} mean reward: {statistics.mean(first):+.3f}")
    print(f"  last  {len(last)} mean reward: {statistics.mean(last):+.3f}")
    step = max(1, n // 10)
    curve = [statistics.mean(rewards[i : i + step]) for i in range(0, n, step)]
    print("  reward curve (10 buckets): " + " ".join(f"{v:+.2f}" for v in curve))


def _team_main(scenario: str, share: str, episodes: int) -> None:
    """Multi-agent scenarios (sim/team.py) — prints sharing on vs off side by side
    unless --share picks one."""
    from agentic_rl.sim import team

    shares = [True, False] if share == "both" else [share == "on"]
    if scenario == "routing":
        correct = asyncio.run(team.run_routing(episodes=episodes))
        print("scenario=routing (Router pick accuracy; capability-cue prior off)")
        _summarize([float(c) for c in correct])
        return
    for on in shares:
        print(f"scenario={scenario} share={'on' if on else 'off'}")
        if scenario == "transfer":
            _summarize(asyncio.run(team.run_transfer(on, episodes_b=episodes)), window=15)
        else:
            rewards, sim = asyncio.run(team.run_conflict(on, rounds=episodes))
            for agent_id, series in rewards.items():
                print(f" agent {agent_id}:")
                _summarize(series, window=30)
            for row in sim.hub.trust_matrix():
                print(f"  trust {row['agent_id']} -> {row['peer_id']}: {row['trust']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic-rl learning simulator.")
    parser.add_argument("--policy", choices=["linucb", "epsilon", "greedy"], default="linucb")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=["single", "transfer", "conflict", "routing"],
        default="single",
        help="single = one agent (the original simulator); the rest are multi-agent (sim/team.py)",
    )
    parser.add_argument("--share", choices=["on", "off", "both"], default="both")
    args = parser.parse_args()

    if args.scenario != "single":
        _team_main(args.scenario, args.share, args.episodes)
        return

    rewards, memory = asyncio.run(run_simulation(args.policy, args.episodes, args.seed))
    print(f"policy={args.policy}")
    _summarize(rewards)

    rules = memory.active_rules()
    print(f"  active rules: {len(rules)}")
    for rule in rules:
        print(f"    (support={rule.support_count}) {rule.text}")


if __name__ == "__main__":
    main()
