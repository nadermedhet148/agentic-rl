from __future__ import annotations

from typing import Any

import numpy as np

from agentic_rl.policy.base import Arm, PeerEvidence, Policy
from agentic_rl.policy.features import FEATURE_DIM


class LinUCBPolicy(Policy):
    """Disjoint LinUCB: one (A, b) pair per arm id, score = theta^T x + alpha * sqrt(x^T A^-1 x).

    `explored` is reported as True when the confidence-bonus term changed the winner
    versus picking by mean estimate (theta^T x) alone — i.e. whether the pick actually
    relied on exploration rather than just agreeing with it.

    Team pooling (docs/MULTI-AGENT-PLAN.md): LinUCB's sufficient statistics are
    additive — a peer's evidence for an arm is (A_j - I, b_j), and adding it is exactly
    what this agent would have learned from seeing the peer's data itself. Only the
    local (A, b) is ever updated or persisted; the trust-weighted peer sum is kept
    separately and replaced wholesale on each `set_peer_evidence`, and `select` scores
    with local + peer.
    """

    id = "linucb"

    def __init__(self, dim: int = FEATURE_DIM, alpha: float = 1.0):
        self._dim = dim
        self._alpha = alpha
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._peer_A: dict[str, np.ndarray] = {}
        self._peer_b: dict[str, np.ndarray] = {}

    def _get(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        if arm_id not in self._A:
            self._A[arm_id] = np.eye(self._dim)
            self._b[arm_id] = np.zeros(self._dim)
        return self._A[arm_id], self._b[arm_id]

    def _effective(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        a, b = self._get(arm_id)
        if arm_id in self._peer_A:
            return a + self._peer_A[arm_id], b + self._peer_b[arm_id]
        return a, b

    def scores(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(mean estimates, UCB scores) per arm, using local + pooled peer evidence.
        Exposed for core/router.py, which adds its own prior on top."""
        mask = explore_mask or [True] * len(arms)
        means = np.empty(len(arms))
        scores = np.empty(len(arms))
        for i, arm in enumerate(arms):
            a, b = self._effective(arm.id)
            a_inv = np.linalg.inv(a)
            theta = a_inv @ b
            mean = float(theta @ arm.features)
            bonus = self._alpha * float(np.sqrt(arm.features @ a_inv @ arm.features)) if mask[i] else 0.0
            means[i] = mean
            scores[i] = mean + bonus
        return means, scores

    def select(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[int, bool]:
        if not arms:
            raise ValueError("select() requires at least one arm")
        means, scores = self.scores(arms, explore_mask)
        chosen = int(np.argmax(scores))
        greedy = int(np.argmax(means))
        return chosen, chosen != greedy

    def update(self, arm: Arm, reward: float) -> None:
        a, b = self._get(arm.id)
        a += np.outer(arm.features, arm.features)
        b += reward * arm.features

    def predict(self, arm: Arm) -> float | None:
        if arm.id not in self._A:
            return None
        a, b = self._A[arm.id], self._b[arm.id]
        return float(np.linalg.solve(a, b) @ arm.features)

    def evidence(self) -> dict[str, Any]:
        eye = np.eye(self._dim)
        return {
            "kind": self.id,
            "dim": self._dim,
            "arms": {
                arm_id: {"A": (self._A[arm_id] - eye).tolist(), "b": self._b[arm_id].tolist()}
                for arm_id in self._A
            },
        }

    def set_peer_evidence(self, peers: list[PeerEvidence]) -> None:
        peer_A: dict[str, np.ndarray] = {}
        peer_b: dict[str, np.ndarray] = {}
        for peer in peers:
            if peer.evidence.get("kind") != self.id or peer.evidence.get("dim") != self._dim:
                continue  # can't pool across policy kinds or feature spaces
            for arm_id, stats in peer.evidence.get("arms", {}).items():
                w = peer.weight(arm_id)
                if w <= 0.0:
                    continue
                peer_A[arm_id] = peer_A.get(arm_id, np.zeros((self._dim, self._dim))) + w * np.asarray(stats["A"])
                peer_b[arm_id] = peer_b.get(arm_id, np.zeros(self._dim)) + w * np.asarray(stats["b"])
        self._peer_A = peer_A
        self._peer_b = peer_b

    def state_dict(self) -> dict[str, Any]:
        return {
            "dim": self._dim,
            "alpha": self._alpha,
            "arms": {
                arm_id: {"A": self._A[arm_id].tolist(), "b": self._b[arm_id].tolist()}
                for arm_id in self._A
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._dim = state.get("dim", self._dim)
        self._alpha = state.get("alpha", self._alpha)
        self._A = {}
        self._b = {}
        for arm_id, arm_state in state.get("arms", {}).items():
            self._A[arm_id] = np.array(arm_state["A"], dtype=np.float64)
            self._b[arm_id] = np.array(arm_state["b"], dtype=np.float64)
