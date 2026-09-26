from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rl.capabilities.base import Outcome, Tier

__all__ = [
    "DEFAULT_AGENT_ID",
    "Action",
    "AgentProfile",
    "Candidate",
    "Episode",
    "Feedback",
    "Memory",
    "Outcome",
    "Session",
    "State",
    "Step",
    "Tier",
]


# The implicit agent every pre-multi-agent episode, rule and policy row belongs to —
# with no agents configured, the whole system is this one agent (docs/MULTI-AGENT-PLAN.md).
DEFAULT_AGENT_ID = "default"


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


class AgentProfile(BaseModel):
    """Identity and specialization of one agent in a team (docs/MULTI-AGENT-PLAN.md).
    The default profile — every capability, no persona — is exactly the single agent
    the system had before multi-agent support."""

    id: str = DEFAULT_AGENT_ID
    name: str = "Default agent"
    role: str = ""  # short label, e.g. "researcher"; shown to the router and to peers
    persona: str = ""  # prepended to the planner prompt
    capabilities: list[str] | None = None  # None = every registered capability; `answer` is always included
    policy: str | None = None  # linucb | epsilon | greedy; None = settings.policy
    share: bool = True  # whether this agent pools knowledge with its peers


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


class Step(BaseModel):
    """One plan -> select -> (confirm) -> execute pass within an episode's loop."""

    index: int
    candidates: list[Candidate]
    action: Action

    outcome: Outcome | None = None  # None while pending_confirmation
    implicit_reward: float = 0.0


class Episode(BaseModel):
    """Full record of one agent-loop run: a request answered by a sequence of steps
    (plan -> select -> confirm-gate -> execute -> observe, repeated) ending either in an
    `answer` step or at settings.max_steps.

    This is both the operational log (drives the reward-weighted policy update) and the
    future fine-tuning dataset (chosen vs. rejected candidates per step) — see rl/export.py.
    """

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)

    state: State
    steps: list[Step] = Field(default_factory=list)
    answer: str | None = None  # set once the `answer` capability executes
    session_id: str | None = None  # groups this episode into a conversation (core/session.py)
    agent_id: str = DEFAULT_AGENT_ID  # which agent in the team ran this episode
    parent_episode_id: str | None = None  # set on a child episode run via delegation

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
    owner_agent_id: str = DEFAULT_AGENT_ID  # the agent whose feedback produced this rule
    scope: Literal["private", "team"] = "team"  # private = shown to its owner only
    text: str  # generalized, imperative: "always send Accept: application/json to api.example.com"
    capability: str | None = None  # scope hint; None = applies to everything

    support_count: int = 1
    source_episode_ids: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    active: bool = True


class Session(BaseModel):
    """A conversation grouping several episodes (see core/session.py:SessionStore).
    Explicitly started/ended by the user (not an always-on notion) — while active,
    its `summary` + the episodes since `summarized_through` are injected into every
    plan (see llm/prompts.py render_conversation()) so later turns can refer back to
    earlier ones (e.g. "now make that a PDF")."""

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    status: Literal["active", "ended"] = "active"
    turn_count: int = 0  # completed episodes attached to this session
    summary: str = ""  # rolling summary of turns older than summarized_through
    summarized_through: int = 0  # turn_count as of the last summarization
