from __future__ import annotations

from abc import ABC, abstractmethod
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
