from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from agentic_rl.core import text as text_util
from agentic_rl.core.models import AgentProfile
from agentic_rl.policy.base import Arm
from agentic_rl.policy.linucb import LinUCBPolicy

ROUTER_DIM = 32
_HASHED = ROUTER_DIM - 2  # dims 0-1 hand-picked, the rest a hashed bag of request tokens/hosts

# Rough cue words for which capability a request needs. Only a *prior* for routing
# before any feedback exists; the bandit's learned estimate takes over as evidence
# accumulates (see Router.select).
_CAPABILITY_CUES: dict[str, re.Pattern[str]] = {
    "http_call": re.compile(r"https?://|\bapi\b|\bendpoint\b|\bpost\b|\bput\b|\bdelete\b", re.I),
    "schedule_task": re.compile(r"\bschedul|\bevery (day|morning|hour|week)|\bremind|\bcron\b|\bdaily\b|\brecurring\b", re.I),
    "web_search": re.compile(r"\bsearch\b|\bwhat is\b|\bwho is\b|\blook up\b|\blatest\b|\bnews\b", re.I),
    "run_code": re.compile(r"\bcalculat|\bcompute\b|\bcode\b|\bpython\b|\bsort\b", re.I),
    "generate_report": re.compile(r"\breport\b|\bpdf\b", re.I),
}


def capability_hints(request: str) -> set[str]:
    return {name for name, pattern in _CAPABILITY_CUES.items() if pattern.search(request)}


def router_features(request: str, source: str) -> np.ndarray:
    vec = np.zeros(ROUTER_DIM, dtype=np.float64)
    vec[0] = 1.0  # bias
    vec[1] = 1.0 if source == "scheduler" else 0.0
    tokens = text_util.tokenize(request) | {f"host:{h}" for h in text_util.extract_hosts(request)}
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        vec[2 + int.from_bytes(digest[:8], "big") % _HASHED] += 1.0
    norm = np.linalg.norm(vec[2:])
    if norm > 0:
        vec[2:] /= norm
    return vec


class Router:
    """Picks which agent in a team handles a request that didn't name one — a
    contextual bandit whose arms are agents (docs/MULTI-AGENT-PLAN.md, routing).

    Reuses LinUCBPolicy unchanged over router_features(); on top of its UCB score it
    adds a capability-cue prior (does this agent have the capability the request
    seems to need?) that fades as the arm accumulates real feedback, so routing is
    sensible from the first request and then learned.
    """

    def __init__(self, profiles: list[AgentProfile], prior_weight: float = 0.5, alpha: float = 1.0):
        self._profiles = {p.id: p for p in profiles}
        self._prior_weight = prior_weight
        self._policy = LinUCBPolicy(dim=ROUTER_DIM, alpha=alpha)
        self._counts: dict[str, int] = {}

    @staticmethod
    def arm_id(agent_id: str) -> str:
        return f"agent:{agent_id}"

    def _arm(self, agent_id: str, features: np.ndarray) -> Arm:
        return Arm(id=self.arm_id(agent_id), features=features, confidence=0.5)

    def hint(self, request: str, profile: AgentProfile) -> float:
        """Fraction of the request's cued capabilities this agent has (a generalist
        with every capability scores 1)."""
        hinted = capability_hints(request)
        if not hinted:
            return 0.0
        if profile.capabilities is None:
            return 1.0
        return len(hinted & set(profile.capabilities)) / len(hinted)

    def select(self, request: str, source: str = "user", explore: bool = True) -> tuple[str, bool]:
        agent_ids = list(self._profiles)
        features = router_features(request, source)
        arms = [self._arm(a, features) for a in agent_ids]
        means, scores = self._policy.scores(arms, [explore] * len(arms))
        prior = np.array(
            [
                self._prior_weight * self.hint(request, self._profiles[a]) / (1 + self._counts.get(a, 0))
                for a in agent_ids
            ]
        )
        chosen = int(np.argmax(scores + prior))
        return agent_ids[chosen], chosen != int(np.argmax(means + prior))

    def update(self, request: str, source: str, agent_id: str, reward: float) -> None:
        if agent_id not in self._profiles:
            return
        self._policy.update(self._arm(agent_id, router_features(request, source)), reward)
        self._counts[agent_id] = self._counts.get(agent_id, 0) + 1

    def state_dict(self) -> dict[str, Any]:
        return {"policy": self._policy.state_dict(), "counts": dict(self._counts)}

    def load_state(self, state: dict[str, Any]) -> None:
        if "policy" in state:
            self._policy.load_state(state["policy"])
        self._counts = dict(state.get("counts", {}))
