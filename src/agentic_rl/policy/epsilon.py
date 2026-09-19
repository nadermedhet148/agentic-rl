from __future__ import annotations

from typing import Any

import numpy as np

from agentic_rl.policy.base import Arm, Policy


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

    def select(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[int, bool]:
        if not arms:
            raise ValueError("select() requires at least one arm")
        mask = explore_mask or [True] * len(arms)
        explorable = [i for i, m in enumerate(mask) if m]

        if explorable and self._rng.random() < self._epsilon:
            return int(self._rng.choice(explorable)), True

        means = [self._means.get(arm.id, 0.5) for arm in arms]
        return int(np.argmax(means)), False

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
