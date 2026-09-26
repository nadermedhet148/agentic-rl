from __future__ import annotations

from typing import Any

import numpy as np

from agentic_rl.policy.base import Arm, PeerEvidence, Policy


class EpsilonGreedyPolicy(Policy):
    """Baseline bandit: tracks a running mean reward per arm id (no context features).
    With probability epsilon, picks uniformly among explorable arms; otherwise picks
    the highest running mean (unseen arms default to an optimistic 0.5)."""

    id = "epsilon"

    def __init__(self, epsilon: float = 0.1, rng: np.random.Generator | None = None):
        self._epsilon = epsilon
        self._rng = rng or np.random.default_rng()
        self._counts: dict[str, int] = {}
        self._means: dict[str, float] = {}
        self._peer_counts: dict[str, float] = {}  # trust-weighted peer counts, pooled
        self._peer_sums: dict[str, float] = {}  # trust-weighted peer count * mean, pooled

    def select(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[int, bool]:
        if not arms:
            raise ValueError("select() requires at least one arm")
        mask = explore_mask or [True] * len(arms)
        explorable = [i for i, m in enumerate(mask) if m]

        if explorable and self._rng.random() < self._epsilon:
            return int(self._rng.choice(explorable)), True

        means = [self._pooled_mean(arm.id) for arm in arms]
        return int(np.argmax(means)), False

    def _pooled_mean(self, arm_id: str) -> float:
        """Count-weighted mean over local + trust-weighted peer observations; the
        optimistic 0.5 prior when nobody has observed this arm."""
        n_local = self._counts.get(arm_id, 0)
        n_peer = self._peer_counts.get(arm_id, 0.0)
        if n_local + n_peer == 0:
            return 0.5
        total = n_local * self._means.get(arm_id, 0.5) + self._peer_sums.get(arm_id, 0.0)
        return total / (n_local + n_peer)

    def predict(self, arm: Arm) -> float | None:
        return self._means.get(arm.id) if self._counts.get(arm.id) else None

    def evidence(self) -> dict[str, Any]:
        return {"kind": self.id, "counts": dict(self._counts), "means": dict(self._means)}

    def set_peer_evidence(self, peers: list[PeerEvidence]) -> None:
        counts: dict[str, float] = {}
        sums: dict[str, float] = {}
        for peer in peers:
            if peer.evidence.get("kind") != self.id:
                continue
            means = peer.evidence.get("means", {})
            for arm_id, n in peer.evidence.get("counts", {}).items():
                w = peer.weight(arm_id)
                if w <= 0.0 or not n:
                    continue
                counts[arm_id] = counts.get(arm_id, 0.0) + w * n
                sums[arm_id] = sums.get(arm_id, 0.0) + w * n * means.get(arm_id, 0.5)
        self._peer_counts = counts
        self._peer_sums = sums

    def update(self, arm: Arm, reward: float) -> None:
        n = self._counts.get(arm.id, 0) + 1
        prev = self._means.get(arm.id, 0.5)
        self._means[arm.id] = prev + (reward - prev) / n
        self._counts[arm.id] = n

    def state_dict(self) -> dict[str, Any]:
        return {"epsilon": self._epsilon, "counts": dict(self._counts), "means": dict(self._means)}

    def load_state(self, state: dict[str, Any]) -> None:
        self._epsilon = state.get("epsilon", self._epsilon)
        self._counts = dict(state.get("counts", {}))
        self._means = dict(state.get("means", {}))
