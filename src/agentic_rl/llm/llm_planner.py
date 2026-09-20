from __future__ import annotations

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # suppress the startup ASCII banner in logs/tests

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from agentic_rl.core import observability
from agentic_rl.core.models import Candidate, State, Step
from agentic_rl.llm import providers
from agentic_rl.llm.base import Planner
from agentic_rl.llm.prompts import (
    SYSTEM_PROMPT,
    render_conversation,
    render_corrections,
    render_history,
    render_rules,
)


class _PlanResponse(BaseModel):
    """Structured-output schema for the planning call — one turn, no tool-call loop
    needed (the planner only *proposes* actions, it never executes them)."""

    candidates: list[Candidate]


class LLMPlanner(Planner):
    """Planner backed by a real LLM via PydanticAI's `Agent` + provider `Model` —
    model-agnostic: `provider` picks claude/openai/google (see llm/providers.py
    for exactly what differs between them and why) and everything else —
    prompt construction, structured-output validation, error mapping, tracing —
    is identical regardless of which provider is selected.

    See docs/ARCHITECTURE.md "LLM calls" for the full design rationale.
    """

    def __init__(
        self,
        provider: str = "claude",
        model: str = "claude-opus-5",
        pydantic_model: Model | None = None,
        history_max_chars: int = 4000,
        max_steps: int = 6,
    ):
        self.id = provider  # instance attr, not class attr — shows up on Episode.planner_id
        self._model_name = model
        self._span_name = f"{provider}.plan"
        self._history_max_chars = history_max_chars
        self._max_steps = max_steps
        resolved_model = pydantic_model or providers.build_model(provider, model)
        self._agent = Agent(
            resolved_model,
            output_type=_PlanResponse,
            instructions=SYSTEM_PROMPT,
            model_settings=providers.build_settings(provider, max_tokens=4096, effort="medium"),
        )

    async def plan(
        self,
        state: State,
        tool_schemas: list[dict],
        prior_corrections: list[str],
        rules: list[str] | None = None,
        history: list[Step] | None = None,
        conversation_summary: str = "",
        conversation_turns: list[tuple[str, str]] | None = None,
    ) -> list[Candidate]:
        history = history or []
        capabilities_block = "\n\n".join(
            f"- {schema['name']}: {schema['description']}\n  input_schema: {schema['input_schema']}"
            for schema in tool_schemas
        )
        user_content = (
            f"Available capabilities:\n{capabilities_block}\n\n"
            f"Request (source={state.source}): {state.request}"
            f"{render_conversation(conversation_summary, conversation_turns or [], self._history_max_chars)}"
            f"{render_history(history, self._history_max_chars, self._max_steps - len(history))}"
            f"{render_rules(rules or [])}"
            f"{render_corrections(prior_corrections)}"
        )

        with observability.generation(self._span_name, model=self._model_name, input=user_content):
            try:
                result = await self._agent.run(user_content)
            except ModelHTTPError as exc:
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                if exc.status_code == 404:
                    raise RuntimeError(f"{self.id} planner: model not found: {exc}") from exc
                if exc.status_code == 429:
                    raise RuntimeError(f"{self.id} planner: rate limited: {exc}") from exc
                raise RuntimeError(f"{self.id} planner: api error ({exc.status_code}): {exc}") from exc
            except ModelAPIError as exc:
                # ModelAPIError is ModelHTTPError's parent — this only catches non-HTTP
                # failures (connection errors) because the more specific case above runs first.
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                raise RuntimeError(f"{self.id} planner: connection error: {exc}") from exc
            except UnexpectedModelBehavior as exc:
                # Structured-output validation exhausted its retries — rare given
                # native JSON-schema enforcement, but not impossible.
                observability.update_current_generation(level="ERROR", status_message=str(exc))
                raise RuntimeError(f"{self.id} planner: unexpected model behavior: {exc}") from exc

            candidates = result.output.candidates
            observability.update_current_generation(
                output=[c.model_dump() for c in candidates],
                usage_details=observability.usage_details(result),
            )

        return candidates
