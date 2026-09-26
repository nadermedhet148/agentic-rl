from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from agentic_rl.capabilities.base import Capability, Outcome, Tier

if TYPE_CHECKING:
    from agentic_rl.core.models import AgentProfile, Episode

# The episode whose step is executing right now — set by core/agent.py around every
# capability.execute call. Capabilities otherwise only see their params; delegation
# also needs to know who is asking and how deep the delegation chain already is.
CURRENT_EPISODE: ContextVar[Episode | None] = ContextVar("agentic_rl_current_episode", default=None)

# (peer agent id, sub-task request, parent episode) -> the peer's child episode
DelegateRunner = Callable[[str, str, "Episode"], Awaitable["Episode"]]


class DelegateCapability(Capability):
    """Hand a sub-task to a peer agent on the team (docs/MULTI-AGENT-PLAN.md,
    delegation). The peer runs it as a child episode, with its own planner, policy,
    capabilities and — crucially — its own safety gates.

    Tier is READ because delegating itself has no side effect: everything the child
    does goes through the child's own confirm gate. If the child pauses for
    confirmation, this returns a `pending` outcome and the parent pauses with it
    (core/agent.py), so delegation can never launder a write past confirmation.
    One instance per agent: the peer list (and so the schema's enum) excludes itself.
    """

    name = "delegate"

    def __init__(self, self_id: str, peers: list[AgentProfile], max_depth: int = 1):
        self._self_id = self_id
        self._peers = {p.id: p for p in peers if p.id != self_id}
        self._max_depth = max_depth
        self._runner: DelegateRunner | None = None
        roster = "; ".join(f"{p.id} ({p.role or p.name})" for p in self._peers.values())
        self.description = (
            "Hand a self-contained sub-task to a peer agent better suited to it, and get "
            f"its answer back. Peers: {roster}."
        )
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "enum": sorted(self._peers), "description": "Which peer."},
                "request": {"type": "string", "description": "The sub-task, stated so the peer needs no other context."},
            },
            "required": ["agent_id", "request"],
            "additionalProperties": False,
        }

    def bind(self, runner: DelegateRunner) -> None:
        """Late-bound because the team (which can run peers) is built after its agents."""
        self._runner = runner

    def tier_for(self, params: dict[str, Any]) -> Tier:
        return Tier.READ

    async def execute(self, params: dict[str, Any]) -> Outcome:
        parent = CURRENT_EPISODE.get()
        peer_id = str(params.get("agent_id", ""))
        request = str(params.get("request", "")).strip()
        if self._runner is None or parent is None:
            return Outcome(ok=False, error="delegation is not available outside a team")
        if peer_id not in self._peers:
            return Outcome(ok=False, error=f"unknown peer agent: {peer_id!r}")
        if not request:
            return Outcome(ok=False, error="request is required")
        if parent.delegation_depth >= self._max_depth:
            return Outcome(ok=False, error=f"delegation depth limit ({self._max_depth}) reached")

        child = await self._runner(peer_id, request, parent)
        payload = {"agent_id": peer_id, "child_episode_id": child.id, "answer": child.answer}
        if child.status == "pending_confirmation":
            return Outcome(ok=True, status="pending_confirmation", payload={**payload, "pending": True})
        return delegation_outcome(child)


def delegation_outcome(child: Episode) -> Outcome:
    """The parent step's outcome once a child episode has finished — used both when
    the child finishes immediately and when it finishes after a confirmation."""
    payload = {"agent_id": child.agent_id, "child_episode_id": child.id, "answer": child.answer, "pending": False}
    if child.status == "executed" and child.answer:
        return Outcome(ok=True, status="200", payload=payload)
    return Outcome(ok=False, payload=payload, error=f"peer agent {child.agent_id!r} finished without an answer")
