from __future__ import annotations

import os
from abc import ABC, abstractmethod

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # suppress the startup ASCII banner in logs/tests

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from agentic_rl.core import observability
from agentic_rl.llm import providers
from agentic_rl.llm.prompts import SUMMARIZER_SYSTEM_PROMPT, render_turns_for_summary


class SummaryResult(BaseModel):
    """What a Summarizer produces from a batch of conversation turns — see
    core/agent.py:_finish_episode, which folds this into Session.summary."""

    summary: str


class Summarizer(ABC):
    """Maintains a rolling summary of a conversation session (core/session.py) so
    the planner prompt doesn't grow unboundedly over a long session — see
    llm/prompts.py render_conversation()."""

    id: str

    @abstractmethod
    async def summarize(self, prior_summary: str, turns: list[tuple[str, str]]) -> str:
        """`turns` are (request, answer) pairs, oldest first. Returns the updated
        summary — replaces `prior_summary` entirely, it isn't appended to by the
        caller."""
        ...


class LLMSummarizer(Summarizer):
    """Summarizer backed by a real LLM via PydanticAI's `Agent` + provider `Model`
    — same model-agnostic pattern as llm/llm_planner.py:LLMPlanner and
    llm/distiller.py:LLMDistiller (see there, and llm/providers.py). Summarization
    happens only every `settings.session_summarize_every` turns, so this runs at
    low effort and infrequently.
    """

    def __init__(
        self,
        provider: str = "claude",
        model: str = "claude-opus-5",
        pydantic_model: Model | None = None,
    ):
        self.id = provider  # instance attr, not class attr — mirrors LLMDistiller
        self._model_name = model
        self._span_name = f"{provider}.summarize"
        resolved_model = pydantic_model or providers.build_model(provider, model)
        self._agent = Agent(
            resolved_model,
            output_type=SummaryResult,
            instructions=SUMMARIZER_SYSTEM_PROMPT,
            model_settings=providers.build_settings(provider, max_tokens=1024, effort="low"),
        )

    async def summarize(self, prior_summary: str, turns: list[tuple[str, str]]) -> str:
        user_content = (
            f"Prior summary: {prior_summary or '(none yet)'}\n\n"
            f"New turns to fold in:\n{render_turns_for_summary(turns)}"
        )

        with observability.generation(self._span_name, model=self._model_name, input=user_content):
            try:
                result = await self._agent.run(user_content)
            except ModelHTTPError as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                if exc.status_code == 404:
                    raise RuntimeError(f"{self.id} summarizer: model not found: {exc}") from exc
                if exc.status_code == 429:
                    raise RuntimeError(f"{self.id} summarizer: rate limited: {exc}") from exc
                raise RuntimeError(f"{self.id} summarizer: api error ({exc.status_code}): {exc}") from exc
            except ModelAPIError as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                raise RuntimeError(f"{self.id} summarizer: connection error: {exc}") from exc
            except UnexpectedModelBehavior as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                raise RuntimeError(f"{self.id} summarizer: unexpected model behavior: {exc}") from exc

            result_output = result.output
            observability.update_current_generation(
                output=result_output.model_dump(),
                usage_details=observability.usage_details(result),
            )

        return result_output.summary


class MockSummarizer(Summarizer):
    """Deterministic summarizer — no network, no API key. Used by tests and the
    simulator. Not a real summary: just folds the new turns onto the prior one as
    text, truncated, so tests can assert it actually ran without needing an LLM."""

    id = "mock"

    async def summarize(self, prior_summary: str, turns: list[tuple[str, str]]) -> str:
        joined = "; ".join(f"{request} -> {answer}" for request, answer in turns)
        combined = f"{prior_summary} {joined}".strip()
        return combined[:2000]
