from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rl.capabilities.base import Outcome, Tier

__all__ = ["Action", "Candidate", "Episode", "Feedback", "Memory", "Outcome", "State", "Tier"]


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


class State(BaseModel):
    """Context the planner and policy condition on for one request."""

    request: str
    source: Literal["user", "scheduler"] = "user"
    intent: str | None = None  # filled in by the planner (a short category label)
    hour_of_day: int = Field(default_factory=lambda: _now().hour, ge=0, le=23)
    prior_correction_count: int = 0  # corrections previously logged for a similar intent


class Candidate(BaseModel):
    """One action the planner proposes for the current state."""

    capability: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_confirmation: bool = False


class Action(BaseModel):
    """The candidate the policy selected, plus whether it was an exploration pick."""

    candidate: Candidate
    index: int  # index into the candidate list this was chosen from
    explored: bool
    arm_id: str  # (capability, param_template_hash, needs_confirmation) — see policy/features.py


class Feedback(BaseModel):
    """User feedback on a completed episode."""

    episode_id: str
    score: Literal[-1, 0, 1]  # thumbs down / neutral / thumbs up
    correction: str | None = None


class Episode(BaseModel):
    """Full record of one agent-loop pass: what was asked, proposed, chosen, and how it went.

    This is both the operational log (drives the reward-weighted policy update) and the
    future fine-tuning dataset (chosen vs. rejected candidates) — see rl/export.py.
    """

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)

    state: State
    candidates: list[Candidate]
    action: Action

    outcome: Outcome | None = None
    status: Literal["pending_confirmation", "executed", "cancelled"] = "executed"

    implicit_reward: float = 0.0
    explicit_score: int | None = None
    correction: str | None = None
    final_reward: float | None = None

    planner_id: str = "unknown"
    policy_id: str = "unknown"


class Memory(BaseModel):
    """A standing rule distilled from one or more corrections (see core/memory.py),
    or added directly by the user. Injected into every plan — see llm/prompts.py
    render_rules() — instead of being fished out per-request by keyword search."""

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    kind: Literal["rule"] = "rule"  # room for "fact"/"preference" later
    text: str  # generalized, imperative: "always send Accept: application/json to api.example.com"
    capability: str | None = None  # scope hint; None = applies to everything

    support_count: int = 1
    source_episode_ids: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    active: bool = True
