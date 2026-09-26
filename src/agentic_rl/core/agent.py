from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agentic_rl.capabilities.answer import AnswerCapability
from agentic_rl.capabilities.base import Outcome, Tier
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.core import observability
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import Action, AgentProfile, Candidate, Episode, Feedback, State, Step
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.base import Planner
from agentic_rl.llm.summarizer import Summarizer
from agentic_rl.policy import features
from agentic_rl.policy.base import Arm, Policy
from agentic_rl.rl import reward as reward_mod

EventCallback = Callable[[str, BaseModel], None]

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
        "steps": len(episode.steps),
        "capabilities": [s.action.candidate.capability for s in episode.steps],
        "answer": (episode.answer or "")[:200],
        "reward": episode.final_reward if episode.final_reward is not None else episode.implicit_reward,
    }


class Agent:
    """Wires planner, policy, capabilities, and the episode store into the loop
    described in docs/PLAN.md: plan -> select -> (confirm gate) -> execute -> observe,
    repeated per episode until an `answer` step or settings.max_steps.
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
        sessions: SessionStore,
        summarizer: Summarizer,
        profile: AgentProfile | None = None,
    ):
        self._profile = profile or AgentProfile()
        if registry.get_or_none("answer") is None:
            registry.register(AnswerCapability())
        if self._profile.capabilities is not None:
            # this agent's slice of the team's capabilities; `answer` is always in it,
            # since every agent must be able to finish an episode
            registry = registry.view([*self._profile.capabilities, "answer"])
        self._planner = planner
        self._policy = policy
        self._registry = registry
        self._store = store
        self._settings = settings
        self._memory = memory
        self._consolidator = consolidator
        self._sessions = sessions
        self._summarizer = summarizer

    @property
    def id(self) -> str:
        return self._profile.id

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    @property
    def policy(self) -> Policy:
        return self._policy

    async def run(
        self,
        request: str,
        source: str = "user",
        episode_id: str | None = None,
        on_event: EventCallback | None = None,
        session_id: str | None = None,
    ) -> Episode:
        with observability.span("agent.run", input=request, source=source, mode=self._settings.mode.value):
            if source == "user":
                self._store.maybe_penalize_reissue(request, source)

            state = State(
                request=request,
                source=source,
                prior_correction_count=self._store.correction_count(request),
            )
            corrections = self._store.search_corrections(request, limit=self._settings.corrections_top_k)
            rules = [m.text for m in self._memory.active_rules(agent_id=self.id)]
            conversation_summary, conversation_turns, resolved_session_id = self._session_context(session_id)

            resolved_episode_id = episode_id or observability.get_current_trace_id()
            episode_kwargs: dict[str, Any] = {
                "state": state,
                "planner_id": self._planner.id,
                "policy_id": self._policy.id,
                "session_id": resolved_session_id,
                "agent_id": self.id,
            }
            if resolved_episode_id:
                episode_kwargs["id"] = resolved_episode_id
            episode = Episode(**episode_kwargs)

            await self._advance(episode, corrections, rules, on_event, conversation_summary, conversation_turns)
            observability.update_current_span(output=_span_output(episode))
            return episode

    def _session_context(self, session_id: str | None) -> tuple[str, list[tuple[str, str]], str | None]:
        """Resolves an active session's conversation-so-far context. A stale/unknown/
        ended session_id from the client is silently ignored — the episode just runs
        session-less, same as if none was ever sent."""
        if not session_id:
            return "", [], None
        session = self._sessions.get(session_id)
        if session is None or session.status != "active":
            return "", [], None
        unsummarized = session.turn_count - session.summarized_through
        recent = self._store.list_session_episodes(session_id)[-unsummarized:] if unsummarized > 0 else []
        conversation_turns = [(e.state.request, e.answer or "") for e in recent]
        return session.summary, conversation_turns, session_id

    async def confirm(self, episode_id: str, on_event: EventCallback | None = None) -> Episode:
        with observability.span("agent.confirm", input=episode_id, trace_id=episode_id):
            episode = self._store.get(episode_id)
            if episode is None:
                raise KeyError(f"unknown episode: {episode_id}")
            if episode.status != "pending_confirmation":
                raise ValueError(f"episode {episode_id} is not pending confirmation (status={episode.status})")

            pending = episode.steps[-1]
            arm = self._arm_for(episode.state, pending.action.candidate)
            done = await self._execute_step(episode, pending, arm, on_event)
            if not done:
                corrections = self._store.search_corrections(
                    episode.state.request, limit=self._settings.corrections_top_k
                )
                rules = [m.text for m in self._memory.active_rules(agent_id=self.id)]
                conversation_summary, conversation_turns, _ = self._session_context(episode.session_id)
                await self._advance(
                    episode, corrections, rules, on_event, conversation_summary, conversation_turns
                )
            observability.update_current_span(output=_span_output(episode))
            return episode

    async def record_feedback(self, feedback: Feedback) -> Episode:
        with observability.span(
            "agent.feedback",
            input=feedback.episode_id,
            trace_id=feedback.episode_id,
            score=feedback.score,
            has_correction=bool(feedback.correction),
        ):
            episode = self._store.apply_feedback(feedback)
            weighted = reward_mod.weighted_reward(episode.explicit_score, episode.correction, episode.implicit_reward)
            for step in episode.steps:
                self._policy.update(self._arm_for(episode.state, step.action.candidate), weighted)
            self._persist_policy()
            if feedback.correction:
                await self._consolidator.consolidate(feedback.correction, episode)
            observability.update_current_span(output=_span_output(episode))
            return episode

    def cancel_task(self, job_id: str) -> Episode | None:
        """Penalize the episode that scheduled `job_id`. Actually removing the job from
        the scheduler is the caller's responsibility (see scheduler/scheduler.py)."""
        return self._store.mark_task_cancelled(job_id)

    async def _advance(
        self,
        episode: Episode,
        corrections: list[str],
        rules: list[str],
        on_event: EventCallback | None,
        conversation_summary: str = "",
        conversation_turns: list[tuple[str, str]] | None = None,
    ) -> None:
        """Plan -> select -> (confirm gate) -> execute -> observe, looped until an
        `answer` step executes, the planner has nothing left to propose, or
        settings.max_steps is reached. Mutates and saves `episode` in place; returns
        (without marking it executed) if a step needs confirmation."""
        while len(episode.steps) < self._settings.max_steps:
            with observability.span("agent.step", input=episode.state.request, index=len(episode.steps)):
                candidates = await self._planner.plan(
                    episode.state,
                    self._registry.tool_schemas(),
                    corrections,
                    rules,
                    history=episode.steps,
                    conversation_summary=conversation_summary,
                    conversation_turns=conversation_turns,
                )
                if not candidates:
                    if episode.steps:
                        break  # planner has nothing left to propose; stop with what we have
                    candidates = [_NO_CANDIDATE.model_copy(deep=True)]
                if not episode.steps:
                    episode.state.intent = candidates[0].capability or None

                arms = [self._arm_for(episode.state, c) for c in candidates]
                explore_mask = [self._explore_allowed(c) for c in candidates]
                idx, explored = self._policy.select(arms, explore_mask)
                chosen = candidates[idx]
                action = Action(candidate=chosen, index=idx, explored=explored, arm_id=arms[idx].id)

                capability = self._registry.get_or_none(chosen.capability)
                tier = capability.tier_for(chosen.params) if capability is not None else Tier.WRITE
                needs_confirm = self._needs_confirmation(chosen, tier, capability_known=capability is not None)

                step = Step(index=len(episode.steps), candidates=candidates, action=action)
                episode.steps.append(step)

                if needs_confirm:
                    episode.status = "pending_confirmation"
                    self._store.save(episode)
                    if on_event:
                        on_event("step", step)
                    return

                done = await self._execute_step(episode, step, arms[idx], on_event)
                if done:
                    return

        episode.status = "executed"
        await self._finish_episode(episode)

    async def _execute_step(
        self, episode: Episode, step: Step, arm: Arm, on_event: EventCallback | None
    ) -> bool:
        """Execute one step's action, finalize its reward, and — if it was the
        terminal `answer` capability — set the episode's answer/status. Returns True
        if the episode is now complete."""
        capability = self._registry.get_or_none(step.action.candidate.capability)
        outcome = await self._execute(capability, step.action.candidate)
        self._finalize_step(episode, step, outcome, arm)
        if on_event:
            on_event("step", step)

        if step.action.candidate.capability == "answer" and outcome.ok:
            payload = outcome.payload
            episode.answer = str(payload.get("text", "")) if isinstance(payload, dict) else ""
            episode.status = "executed"
            await self._finish_episode(episode)
            return True
        return False

    async def _finish_episode(self, episode: Episode) -> None:
        """Saves a just-completed episode and, if it belongs to an active session,
        bumps that session's turn_count and folds completed turns into the rolling
        summary every settings.session_summarize_every turns (core/session.py)."""
        self._store.save(episode)
        if not episode.session_id:
            return
        session = self._sessions.get(episode.session_id)
        if session is None or session.status != "active":
            return

        turn_count = session.turn_count + 1
        unsummarized = turn_count - session.summarized_through
        if unsummarized >= self._settings.session_summarize_every:
            try:
                turns = self._store.list_session_episodes(episode.session_id)[-unsummarized:]
                pairs = [(e.state.request, e.answer or "") for e in turns]
                new_summary = await self._summarizer.summarize(session.summary, pairs)
            except Exception:  # noqa: BLE001 - the user already has their answer; never fail the response for this
                self._sessions.update(episode.session_id, turn_count=turn_count)
            else:
                self._sessions.update(
                    episode.session_id, turn_count=turn_count, summary=new_summary, summarized_through=turn_count
                )
        else:
            self._sessions.update(episode.session_id, turn_count=turn_count)

    def _finalize_step(self, episode: Episode, step: Step, outcome: Outcome, arm: Arm) -> None:
        step.outcome = outcome
        step.implicit_reward = reward_mod.implicit_reward(executed_ok=outcome.ok)
        finalized = [s.implicit_reward for s in episode.steps if s.outcome is not None]
        episode.implicit_reward = sum(finalized) / len(finalized)
        episode.final_reward = episode.implicit_reward
        self._store.save(episode)
        self._policy.update(arm, reward_mod.weighted_reward(None, None, step.implicit_reward))
        self._persist_policy()

    def _persist_policy(self) -> None:
        """Save the policy's learned state after every update — see policy/base.py
        state_dict()/load_state() and api/app.py where it's reloaded on boot. Cheap:
        one row, one update per episode."""
        self._store.save_policy_state(self._policy.id, self._policy.state_dict(), agent_id=self.id)

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
        with observability.span("capability.execute", input=candidate.params, capability=candidate.capability):
            if capability is None:
                outcome = Outcome(ok=False, error=f"unknown capability: {candidate.capability!r}")
            else:
                try:
                    outcome = await capability.execute(candidate.params)
                except Exception as exc:  # noqa: BLE001 - a capability bug must not crash the loop
                    outcome = Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")
            observability.update_current_span(
                output={"ok": outcome.ok, "status": outcome.status, "error": outcome.error}
            )
            return outcome
