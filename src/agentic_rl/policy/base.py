from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Arm:
    """A candidate as seen by the policy: its bandit identity, feature vector, and
    the planner's own confidence (used by the Greedy baseline)."""

    id: str
    features: np.ndarray
    confidence: float


@dataclass
class PeerEvidence:
    """One peer's locally learned evidence, as handed to `Policy.set_peer_evidence`
    by core/hub.py:KnowledgeHub. `evidence` is exactly the peer's `Policy.evidence()`
    — JSON-serializable, so it could equally have arrived over the network from an
    agent in another process. `weight(arm_id)` is how much the receiving agent
    trusts this peer about that arm, in [0, 1]."""

    agent_id: str
    evidence: dict[str, Any]
    weight: Callable[[str], float]


class Policy(ABC):
    """Chooses among the planner's candidates and learns from observed rewards.

    `explore_mask[i]`, when provided to `select`, is False for any arm that must not
    be explored right now (e.g. a write-tier action in prod-strict mode) — the safety
    rule lives here so it's enforced by construction rather than by callers remembering
    to gate it.
    """

    id: str

    @abstractmethod
    def select(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[int, bool]:
        """Return (index into `arms` that was chosen, whether that pick was exploratory)."""
        ...

    @abstractmethod
    def update(self, arm: Arm, reward: float) -> None:
        """Incorporate an observed reward for the arm that was chosen."""
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of everything this policy has learned, for
        persistence (see core/store.py: policy_state table)."""
        ...

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by `state_dict()`. Called once, right after
        construction, before any `select`/`update` — see api/app.py:create_app."""
        ...

    # --- knowledge sharing (docs/MULTI-AGENT-PLAN.md, procedural sharing) -----------
    # Non-abstract on purpose: a policy with nothing to share (GreedyPolicy) simply
    # keeps these defaults, and pooling becomes a no-op for it.

    def evidence(self) -> dict[str, Any]:
        """This policy's *locally observed* evidence only (never pooled peer
        evidence, so re-sharing can't double-count), JSON-serializable. Must carry a
        "kind" key; peers only pool evidence of their own kind."""
        return {"kind": self.id}

    def set_peer_evidence(self, peers: list[PeerEvidence]) -> None:
        """Replace (never accumulate) the pooled peer evidence `select` scores with.
        Replacing makes pooling idempotent: syncing twice is the same as once."""

    def predict(self, arm: Arm) -> float | None:
        """This policy's own-evidence-only mean reward estimate for `arm`, or None if
        it has never observed that arm. Used by KnowledgeHub to learn trust: how well
        does agent j's model predict the rewards agent i actually gets?"""
        return None
