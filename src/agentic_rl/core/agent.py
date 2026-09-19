from __future__ import annotations

from agentic_rl.capabilities.base import Outcome, Tier
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.core import observability
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import Action, Candidate, Episode, Feedback, State
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.base import Planner
from agentic_rl.policy import features
from agentic_rl.policy.base import Arm, Policy
from agentic_rl.rl import reward as reward_mod

_NO_CANDIDATE = Candidate(
    capability="",
    params={},
    rationale="planner returned no candidates",
    confidence=0.0,
    needs_confirmation=True,
)


def _span_output(episode: Episode) -> dict:
    """Compact trace-output summary for an episode — used at every agent.* span's
    exit point (see core/observability.py)."""
    return {
        "episode_id": episode.id,
        "status": episode.status,
        "capability": episode.action.candidate.capability,
        "explored": episode.action.explored,
        "outcome_ok": episode.outcome.ok if episode.outcome else None,
        "reward": episode.final_reward if episode.final_reward is not None else episode.implicit_reward,
    }


class Agent:
    """Wires planner, policy, capabilities, and the episode store into the loop
    described in docs/PLAN.md: plan -> select -> (confirm gate) -> execute -> reward.
    """

    def __init__(
        self,
        planner: Planner,
        policy: Policy,
        registry: CapabilityRegistry,
        store: EpisodeStore,
        settings: Settings,
        memory: MemoryStore,
        consolidator: Consolidator,
    ):
        self._planner = planner
        self._policy = policy
        self._registry = registry
        self._store = store
        self._settings = settings
        self._memory = memory
        self._consolidator = consolidator

    async def run(self, request: str, source: str = "user") -> Episode:
        with observability.span("agent.run", input=request, source=source, mode=self._settings.mode.value):
            if source == "user":
                self._store.maybe_penalize_reissue(request, source)

            state = State(
                request=request,
                source=source,
                prior_correction_count=self._store.correction_count(request),
            )
            corrections = self._store.search_corrections(request, limit=self._settings.corrections_top_k)
            rules = [m.text for m in self._memory.active_rules()]
            candidates = await self._planner.plan(state, self._registry.tool_schemas(), corrections, rules)
            if not candidates:
                candidates = [_NO_CANDIDATE.model_copy(deep=True)]
            state.intent = candidates[0].capability or None

            arms = [self._arm_for(state, c) for c in candidates]
            explore_mask = [self._explore_allowed(c) for c in candidates]
            idx, explored = self._policy.select(arms, explore_mask)
            chosen = candidates[idx]
            action = Action(candidate=chosen, index=idx, explored=explored, arm_id=arms[idx].id)

            capability = self._registry.get_or_none(chosen.capability)
            tier = capability.tier_for(chosen.params) if capability is not None else Tier.WRITE
            needs_confirm = self._needs_confirmation(chosen, tier, capability_known=capability is not None)

            episode = Episode(
                state=state,
                candidates=candidates,
                action=action,
                outcome=None,
                status="pending_confirmation" if needs_confirm else "executed",
                implicit_reward=0.0,
                planner_id=self._planner.id,
                policy_id=self._policy.id,
            )

            if needs_confirm:
                self._store.save(episode)
                observability.update_current_span(output=_span_output(episode))
                return episode

            outcome = await self._execute(capability, chosen)
            self._finalize_execution(episode, outcome, arms[idx])
            observability.update_current_span(output=_span_output(episode))
            return episode

    async def confirm(self, episode_id: str) -> Episode:
        with observability.span("agent.confirm", input=episode_id):
            episode = self._store.get(episode_id)
            if episode is None:
                raise KeyError(f"unknown episode: {episode_id}")
            if episode.status != "pending_confirmation":
                raise ValueError(f"episode {episode_id} is not pending confirmation (status={episode.status})")

            capability = self._registry.get_or_none(episode.action.candidate.capability)
            outcome = await self._execute(capability, episode.action.candidate)
            arm = self._arm_for(episode.state, episode.action.candidate)
            self._finalize_execution(episode, outcome, arm)
            observability.update_current_span(output=_span_output(episode))
            return episode

    async def record_feedback(self, feedback: Feedback) -> Episode:
        with observability.span(
            "agent.feedback", input=feedback.episode_id, score=feedback.score, has_correction=bool(feedback.correction)
        ):
            episode = self._store.apply_feedback(feedback)
            arm = self._arm_for(episode.state, episode.action.candidate)
            self._policy.update(
                arm,
                reward_mod.weighted_reward(episode.explicit_score, episode.correction, episode.implicit_reward),
            )
            self._persist_policy()
            if feedback.correction:
                await self._consolidator.consolidate(feedback.correction, episode)
            observability.update_current_span(output=_span_output(episode))
            return episode

    def cancel_task(self, job_id: str) -> Episode | None:
        """Penalize the episode that scheduled `job_id`. Actually removing the job from
        the scheduler is the caller's responsibility (see scheduler/scheduler.py)."""
        return self._store.mark_task_cancelled(job_id)

    def _finalize_execution(self, episode: Episode, outcome: Outcome, arm: Arm) -> None:
        episode.outcome = outcome
        episode.status = "executed"
        episode.implicit_reward = reward_mod.implicit_reward(executed_ok=outcome.ok)
        episode.final_reward = episode.implicit_reward
        self._store.save(episode)
        self._policy.update(arm, reward_mod.weighted_reward(None, None, episode.implicit_reward))
        self._persist_policy()

    def _persist_policy(self) -> None:
        """Save the policy's learned state after every update — see policy/base.py
        state_dict()/load_state() and api/app.py where it's reloaded on boot. Cheap:
        one row, one update per episode."""
        self._store.save_policy_state(self._policy.id, self._policy.state_dict())

    def _arm_for(self, state: State, candidate: Candidate) -> Arm:
        return Arm(
            id=features.arm_id(candidate),
            features=features.build_features(
                state,
                candidate,
                capability_success_rate=self._store.capability_success_rate(candidate.capability),
                correction_count=state.prior_correction_count,
            ),
            confidence=candidate.confidence,
        )

    def _explore_allowed(self, candidate: Candidate) -> bool:
        if self._settings.mode == Mode.SIM:
            return True
        if self._settings.mode == Mode.PROD_STRICT:
            return False
        capability = self._registry.get_or_none(candidate.capability)
        tier = capability.tier_for(candidate.params) if capability is not None else Tier.WRITE
        return tier == Tier.READ

    def _needs_confirmation(self, candidate: Candidate, tier: Tier, capability_known: bool) -> bool:
        if not capability_known:
            return True  # can't safely auto-execute an unregistered capability
        if tier == Tier.READ:
            return False
        if self._settings.mode == Mode.PROD_STRICT:
            return True  # every write-tier action requires confirmation in prod-strict
        return candidate.needs_confirmation

    async def _execute(self, capability, candidate: Candidate) -> Outcome:
        if capability is None:
            return Outcome(ok=False, error=f"unknown capability: {candidate.capability!r}")
        try:
            return await capability.execute(candidate.params)
        except Exception as exc:  # noqa: BLE001 - a capability bug must not crash the loop
            return Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")
