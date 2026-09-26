from __future__ import annotations

from agentic_rl.core import observability
from agentic_rl.core.agent import Agent, EventCallback
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.hub import KnowledgeHub
from agentic_rl.core.models import AgentProfile, Episode, Feedback
from agentic_rl.core.router import Router
from agentic_rl.core.store import EpisodeStore
from agentic_rl.rl import reward as reward_mod

ROUTER_OWNER_ID = "__team__"  # policy_state row owner for the router's learned state
ROUTER_POLICY_ID = "router"


class Team:
    """Several specialist agents behind one entry point (docs/MULTI-AGENT-PLAN.md).

    - Routing: a request without an explicit agent goes to whichever agent the
      Router picks, and the Router learns from the same feedback the agents do.
    - Delegation: an agent's `delegate` step runs a peer as a child episode; if the
      child pauses for confirmation, the parent pauses too, and confirming either
      one resumes the chain (`confirm`).
    - Credit: feedback on a parent flows to its children at `delegation_credit`.
    - Knowledge sharing itself lives in each Agent's KnowledgeHub hooks.

    With a single agent, this is a thin pass-through to it.
    """

    def __init__(
        self,
        agents: list[Agent],
        store: EpisodeStore,
        settings: Settings,
        hub: KnowledgeHub,
        router: Router | None = None,
    ):
        if not agents:
            raise ValueError("a team needs at least one agent")
        self._agents = {a.id: a for a in agents}
        self._default_id = agents[0].id
        self._store = store
        self._settings = settings
        self._hub = hub
        self._router = router
        if router is not None:
            saved = store.load_policy_state(ROUTER_POLICY_ID, agent_id=ROUTER_OWNER_ID)
            if saved is not None:
                router.load_state(saved)

    # --- lookup ----------------------------------------------------------------------

    @property
    def hub(self) -> KnowledgeHub:
        return self._hub

    @property
    def default_agent(self) -> Agent:
        return self._agents[self._default_id]

    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    def profiles(self) -> list[AgentProfile]:
        return [a.profile for a in self._agents.values()]

    def get(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"unknown agent: {agent_id}") from None

    def route(self, request: str, source: str = "user") -> tuple[str, bool]:
        """(agent id, whether the Router picked it) for a request with no agent named."""
        if self._router is None or len(self._agents) == 1:
            return self._default_id, False
        agent_id, _explored = self._router.select(
            request, source, explore=self._settings.mode != Mode.PROD_STRICT
        )
        return agent_id, True

    # --- the loop ----------------------------------------------------------------------

    async def run(
        self,
        request: str,
        source: str = "user",
        episode_id: str | None = None,
        on_event: EventCallback | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> Episode:
        routed = False
        if agent_id is None:
            agent_id, routed = self.route(request, source)
        agent = self.get(agent_id)
        with observability.span("team.run", input=request, agent_id=agent_id, routed=routed):
            episode = await agent.run(
                request, source=source, episode_id=episode_id, on_event=on_event, session_id=session_id, routed=routed
            )
        self._after_completion(episode)
        return episode

    async def delegate(self, peer_id: str, request: str, parent: Episode) -> Episode:
        """The runner bound into every agent's DelegateCapability."""
        with observability.span("team.delegate", input=request, from_agent=parent.agent_id, to_agent=peer_id):
            return await self.get(peer_id).run(request, source=parent.state.source, parent=parent)

    async def confirm(self, episode_id: str, on_event: EventCallback | None = None) -> Episode:
        episode = self._store.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode: {episode_id}")
        pending = episode.steps[-1] if episode.steps else None
        if (
            episode.status == "pending_confirmation"
            and pending is not None
            and pending.outcome is None
            and pending.child_episode_id
        ):
            # waiting on a delegated child: confirming the parent confirms the child,
            # whose completion resumes this episode (via _resume_parent)
            await self.confirm(pending.child_episode_id)
            refreshed = self._store.get(episode_id)
            return refreshed if refreshed is not None else episode

        episode = await self.get(episode.agent_id).confirm(episode_id, on_event=on_event)
        self._after_completion(episode)
        await self._resume_parent(episode)
        return episode

    async def _resume_parent(self, child: Episode) -> None:
        """If `child` just finished and its parent is paused waiting on it, resume the
        parent — and so on up the delegation chain."""
        if child.status == "pending_confirmation" or not child.parent_episode_id:
            return
        parent = self._store.get(child.parent_episode_id)
        if parent is None or parent.status != "pending_confirmation" or not parent.steps:
            return
        pending = parent.steps[-1]
        if pending.outcome is not None or pending.child_episode_id != child.id:
            return
        parent = await self.get(parent.agent_id).resume_delegation(parent.id, child)
        self._after_completion(parent)
        await self._resume_parent(parent)

    async def record_feedback(self, feedback: Feedback) -> Episode:
        existing = self._store.get(feedback.episode_id)
        if existing is None:
            raise KeyError(f"unknown episode: {feedback.episode_id}")
        episode = await self.get(existing.agent_id).record_feedback(feedback)
        weighted = reward_mod.weighted_reward(episode.explicit_score, episode.correction, episode.implicit_reward)
        if episode.routed:
            self._update_router(episode, weighted)
        self._credit_children(episode, weighted * self._settings.delegation_credit)
        return episode

    def cancel_task(self, job_id: str) -> Episode | None:
        return self.default_agent.cancel_task(job_id)

    # --- learning bookkeeping -------------------------------------------------------------

    def _after_completion(self, episode: Episode) -> None:
        """Implicit routing signal once a routed episode finishes (feedback, if any
        arrives later, is the stronger explicit one — see record_feedback)."""
        if episode.routed and episode.status == "executed":
            self._update_router(episode, reward_mod.weighted_reward(None, None, episode.implicit_reward))

    def _update_router(self, episode: Episode, reward: float) -> None:
        if self._router is None:
            return
        self._router.update(episode.state.request, episode.state.source, episode.agent_id, reward)
        self._store.save_policy_state(ROUTER_POLICY_ID, self._router.state_dict(), agent_id=ROUTER_OWNER_ID)

    def _credit_children(self, episode: Episode, reward: float) -> None:
        for step in episode.steps:
            if not step.child_episode_id:
                continue
            child = self._store.get(step.child_episode_id)
            if child is None or child.agent_id not in self._agents:
                continue
            self._agents[child.agent_id].apply_delegated_credit(child.id, reward)
            self._credit_children(child, reward * self._settings.delegation_credit)
