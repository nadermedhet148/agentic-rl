from __future__ import annotations

from typing import Any

import numpy as np

from agentic_rl.policy.base import Arm, Policy
from agentic_rl.policy.features import FEATURE_DIM


class LinUCBPolicy(Policy):
    """Disjoint LinUCB: one (A, b) pair per arm id, score = theta^T x + alpha * sqrt(x^T A^-1 x).

    `explored` is reported as True when the confidence-bonus term changed the winner
    versus picking by mean estimate (theta^T x) alone — i.e. whether the pick actually
    relied on exploration rather than just agreeing with it.
    """

    id = "linucb"

    def __init__(self, dim: int = FEATURE_DIM, alpha: float = 1.0):
        self._dim = dim
        self._alpha = alpha
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}

    def _get(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        if arm_id not in self._A:
            self._A[arm_id] = np.eye(self._dim)
            self._b[arm_id] = np.zeros(self._dim)
        return self._A[arm_id], self._b[arm_id]

    def select(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[int, bool]:
        if not arms:
            raise ValueError("select() requires at least one arm")
        mask = explore_mask or [True] * len(arms)

        means = np.empty(len(arms))
        scores = np.empty(len(arms))
        for i, arm in enumerate(arms):
            a, b = self._get(arm.id)
            a_inv = np.linalg.inv(a)
            theta = a_inv @ b
            mean = float(theta @ arm.features)
            bonus = self._alpha * float(np.sqrt(arm.features @ a_inv @ arm.features)) if mask[i] else 0.0
            means[i] = mean
            scores[i] = mean + bonus

        chosen = int(np.argmax(scores))
        greedy = int(np.argmax(means))
        return chosen, chosen != greedy

    def update(self, arm: Arm, reward: float) -> None:
        a, b = self._get(arm.id)
        a += np.outer(arm.features, arm.features)
        b += reward * arm.features

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
