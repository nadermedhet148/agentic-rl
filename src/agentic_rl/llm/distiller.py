from __future__ import annotations

import os
from abc import ABC, abstractmethod

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # suppress the startup ASCII banner in logs/tests

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from agentic_rl.core import observability
from agentic_rl.core.models import Episode, Memory
from agentic_rl.llm import providers
from agentic_rl.llm.prompts import DISTILLER_SYSTEM_PROMPT, render_existing_rules


class DistillResult(BaseModel):
    """What a Distiller produces from one correction — see core/memory.py Consolidator,
    which turns this into a Memory (new, bumped, or superseding an existing one)."""

    rule_text: str
    capability: str | None = None
    matches_existing_id: str | None = None
    supersedes_id: str | None = None


class Distiller(ABC):
    """Generalizes a correction into a reusable rule and relates it to the rules
    already on file — matching, contradicting, or neither."""

    id: str

    @abstractmethod
    async def distill(self, correction: str, episode: Episode, existing: list[Memory]) -> DistillResult:
        """`existing` are candidate rules the correction might match or contradict —
        Consolidator only accepts an id from this list; anything else is ignored as
        a hallucinated reference."""
        ...


class LLMDistiller(Distiller):
    """Distiller backed by a real LLM via PydanticAI's `Agent` + provider `Model`
    — model-agnostic, same rationale as llm/llm_planner.py:LLMPlanner (see there,
    and see llm/providers.py, and docs/ARCHITECTURE.md "LLM calls"). Corrections
    are rare (one call per piece of explicit feedback with a correction
    attached), so this runs at low effort and doesn't need the caching-sensitive
    prompt structure the planner does.
    """

    def __init__(
        self,
        provider: str = "claude",
        model: str = "claude-opus-5",
        pydantic_model: Model | None = None,
    ):
        self.id = provider  # instance attr, not class attr — shows up on Episode.planner_id
        self._model_name = model
        self._span_name = f"{provider}.distill"
        resolved_model = pydantic_model or providers.build_model(provider, model)
        self._agent = Agent(
            resolved_model,
            output_type=DistillResult,
            instructions=DISTILLER_SYSTEM_PROMPT,
            model_settings=providers.build_settings(provider, max_tokens=2048, effort="low"),
        )

    async def distill(self, correction: str, episode: Episode, existing: list[Memory]) -> DistillResult:
        actions = "\n".join(
            f"{i + 1}. {s.action.candidate.capability} with params {s.action.candidate.params}"
            for i, s in enumerate(episode.steps)
        )
        user_content = (
            f"Original request: {episode.state.request}\n"
            f"Actions taken:\n{actions}\n"
            f"User's correction: {correction}"
            f"{render_existing_rules([(m.id, m.text) for m in existing])}"
        )

        with observability.generation(self._span_name, model=self._model_name, input=user_content):
            try:
                result = await self._agent.run(user_content)
            except ModelHTTPError as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                if exc.status_code == 404:
                    raise RuntimeError(f"{self.id} distiller: model not found: {exc}") from exc
                if exc.status_code == 429:
                    raise RuntimeError(f"{self.id} distiller: rate limited: {exc}") from exc
                raise RuntimeError(f"{self.id} distiller: api error ({exc.status_code}): {exc}") from exc
            except ModelAPIError as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                raise RuntimeError(f"{self.id} distiller: connection error: {exc}") from exc
            except UnexpectedModelBehavior as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                raise RuntimeError(f"{self.id} distiller: unexpected model behavior: {exc}") from exc

            result_output = result.output
            observability.update_current_generation(
                output=result_output.model_dump(),
                usage_details=observability.usage_details(result),
            )

        return result_output


class MockDistiller(Distiller):
    """Deterministic distiller — no network, no API key. Used by tests and the
    simulator. The rule is the correction text verbatim; it matches an existing rule
    only on an exact normalized (lowercased, whitespace-collapsed) text match, and
    never supersedes — it can't judge contradiction without an LLM."""

    id = "mock"

    async def distill(self, correction: str, episode: Episode, existing: list[Memory]) -> DistillResult:
        normalized = " ".join(correction.lower().split())
        for memory in existing:
            if " ".join(memory.text.lower().split()) == normalized:
                return DistillResult(rule_text=correction, matches_existing_id=memory.id)
        return DistillResult(rule_text=correction)
