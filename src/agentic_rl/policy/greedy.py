from __future__ import annotations

from typing import Any

import numpy as np

from agentic_rl.policy.base import Arm, Policy


class GreedyPolicy(Policy):
    """Control-group baseline: always takes the planner's own top-confidence candidate
    and never learns. Useful to confirm a learning policy actually beats "no RL"."""

    id = "greedy"

    def select(self, arms: list[Arm], explore_mask: list[bool] | None = None) -> tuple[int, bool]:
        if not arms:
            raise ValueError("select() requires at least one arm")
        confidences = [arm.confidence for arm in arms]
        return int(np.argmax(confidences)), False

    def update(self, arm: Arm, reward: float) -> None:
        pass  # by design: this policy never learns

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        pass  # nothing to restore
