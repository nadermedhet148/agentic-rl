from __future__ import annotations

import re
from collections.abc import Callable

from agentic_rl.core.models import Candidate, State, Step
from agentic_rl.llm.base import Planner

Rule = tuple[str, list[Candidate]]  # (substring to match in request, candidates to return)
DefaultFn = Callable[[State, list[Step]], list[Candidate]]

_URL_RE = re.compile(r"https?://\S+")
_SCHEDULE_WORDS = ("schedule", "every day", "every morning", "remind", "cron", "daily", "recurring")
_POST_WORDS = ("post", "create", "send", "submit")
_DELETE_WORDS = ("delete", "remove", "cancel")
_PUT_WORDS = ("put", "update", "replace")


class MockPlanner(Planner):
    """Deterministic planner driven by fixtures — no network, no API key.

    Used by unit tests and the simulator (sim/) so the learning loop can be exercised
    and measured without an LLM in the critical path. Rules are matched by case-insensitive
    substring against `state.request`, first match wins; `default_fn` (or `default`) is used
    otherwise.
    """

    id = "mock"

    def __init__(
        self,
        rules: list[Rule] | None = None,
        default: list[Candidate] | None = None,
        default_fn: DefaultFn | None = None,
    ):
        self._rules = rules or []
        self._default = default or []
        self._default_fn = default_fn

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
        if history:
            # Rules/default match on the initial request text — reapplying them after
            # step 0 would repeat the same candidate forever (until max_steps). Once
            # there's history, hand off to default_fn (if given) or just answer.
            if self._default_fn is not None:
                return self._default_fn(state, history)
            return [_answer_from_history(history)]

        request_lower = state.request.lower()
        for substring, candidates in self._rules:
            if substring.lower() in request_lower:
                return [c.model_copy(deep=True) for c in candidates]
        if self._default_fn is not None:
            return self._default_fn(state, history)
        return [c.model_copy(deep=True) for c in self._default]


def _answer_from_history(history: list[Step]) -> Candidate:
    """A canned `answer` candidate summarizing the last step — the fallback used once
    MockPlanner has history and no `default_fn` was given, so every existing
    single-step test still terminates after one action step + one answer step."""
    last = history[-1]
    candidate = last.action.candidate
    outcome = last.outcome
    if outcome is None:
        text = f"Done: {candidate.capability} is pending confirmation."
    elif outcome.ok:
        text = f"Done: {candidate.capability} succeeded."
    else:
        text = f"Done: {candidate.capability} failed ({outcome.error})."
    return Candidate(
        capability="answer",
        params={"text": text},
        rationale="heuristic mock planner: summarizing the last step",
        confidence=0.9,
    )


def heuristic_default_fn(state: State, history: list[Step]) -> list[Candidate]:
    """A no-LLM stand-in good enough to exercise the app without a Claude API key —
    used as MockPlanner's `default_fn` in api/app.py when AGENTIC_RL_PLANNER=mock
    (the default). Not a real planner: it pattern-matches a URL or scheduling
    language in the request text on the first step, then always answers. Anything
    else on the first step falls back to no candidates, same as a bare MockPlanner,
    which forces a pending_confirmation the user can inspect and reject.
    """
    if history:
        return [_answer_from_history(history)]

    text = state.request
    text_lower = text.lower()

    url_match = _URL_RE.search(text)
    if url_match:
        url = url_match.group(0).rstrip(".,;)")
        method = "GET"
        if any(word in text_lower for word in _POST_WORDS):
            method = "POST"
        elif any(word in text_lower for word in _DELETE_WORDS):
            method = "DELETE"
        elif any(word in text_lower for word in _PUT_WORDS):
            method = "PUT"
        return [
            Candidate(
                capability="http_call",
                params={"method": method, "url": url},
                rationale=f"heuristic mock planner: saw a URL, inferred {method} from wording",
                confidence=0.6,
                needs_confirmation=method != "GET",
            )
        ]

    if any(word in text_lower for word in _SCHEDULE_WORDS):
        return [
            Candidate(
                capability="schedule_task",
                params={"instruction": text, "cron": "0 9 * * *"},
                rationale="heuristic mock planner: saw scheduling language, defaulted to daily 9am UTC",
                confidence=0.5,
                needs_confirmation=True,
            )
        ]

    return []
