from __future__ import annotations

import json

from agentic_rl.core.models import Step

SYSTEM_PROMPT = """\
You are the planning component of an agent that answers requests by executing a small \
set of predefined capabilities, one step at a time. Each call to you proposes only the \
next step: 1-3 candidate actions, ordered most-likely-correct first, given the request \
and — after the first step — the outcomes of the steps already taken.

Each candidate names exactly one capability and its parameters. Build a later step's \
parameters from an earlier step's outcome when the request requires it (e.g. use a \
prior payload's field as this step's input) — you will be shown a summary of every \
prior step and its outcome. Never propose a step identical to one already completed.

Set `needs_confirmation` to true for any action with side effects that would be costly \
or awkward to undo (e.g. a POST/PUT/PATCH/DELETE http_call, or a recurring \
schedule_task) unless the request very explicitly asked for exactly that action. Always \
set `needs_confirmation` to true for `run_code` — it executes real Python on the \
server. Use `run_code` only for computation or data reshaping you can't do reliably \
yourself (exact math, sorting, reformatting), never to fetch a URL (use http_call or \
web_search for that instead). Use `generate_report` once you already have the content \
to report on, to turn it into a PDF — not to gather that content. Set \
`confidence` to your honest estimate (0-1) that this candidate is what the user wants.

Once you have everything needed to respond — including when the request needs no \
capability call at all — propose the `answer` capability with `text` set to the final, \
complete reply to show the user. Never invent capabilities that weren't listed.

You will sometimes be shown corrections from past interactions where a previous \
candidate was wrong. Treat these as binding instructions for this user: do not repeat \
a corrected mistake.

You will sometimes be shown the conversation so far in this session (a summary of \
earlier turns and/or the most recent ones verbatim). The user may refer back to \
something from an earlier turn without repeating it (e.g. "now make that a PDF") — \
use that section to resolve what "that" means before asking the user to repeat \
themselves."""


def render_history(steps: list[Step], max_chars: int, steps_remaining: int | None = None) -> str:
    """Steps already executed in this episode's loop (core/agent.py `_advance`),
    rendered so the planner can chain an outcome into the next step's params and
    knows what's already been tried. Empty until the loop's second call.

    `steps_remaining` is `settings.max_steps - len(steps)`; when it's exactly 1 a
    reminder is appended that this is the last step, since the loop won't call the
    planner again after it.
    """
    if not steps:
        return ""
    lines = ["\n\nSteps completed so far:"]
    for step in steps:
        candidate = step.action.candidate
        outcome = step.outcome
        header = f"step {step.index + 1}: {candidate.capability}({candidate.params})"
        if outcome is None:
            lines.append(f"{header} -> pending confirmation")
        elif outcome.ok:
            payload = json.dumps(outcome.payload, default=str)[:max_chars]
            lines.append(f"{header} -> ok (status {outcome.status})\n  payload: {payload}")
        else:
            lines.append(f"{header} -> error: {outcome.error}")
    if steps_remaining == 1:
        lines.append("\nThis is the last allowed step: you must propose `answer`.")
    return "\n".join(lines)


def render_conversation(summary: str, recent_turns: list[tuple[str, str]], max_chars: int) -> str:
    """The active session's conversation so far (core/session.py:Session), rendered
    so the planner can resolve a reference to an earlier turn (e.g. "make that a
    PDF"). `summary` covers turns already folded in (see llm/summarizer.py);
    `recent_turns` are (request, answer) pairs for turns not yet summarized, oldest
    first — empty/"" when no session is active. Rendered before render_history():
    this is broader, cross-episode context, this episode's own step progress is
    narrower and more specific.
    """
    if not summary and not recent_turns:
        return ""
    parts = ["\n\nConversation so far in this session:"]
    if summary:
        parts.append(f"Summary of earlier turns: {summary}")
    for request, answer in recent_turns:
        parts.append(f"User: {request[:max_chars]}\nAgent: {answer[:max_chars]}")
    return "\n".join(parts)


def render_corrections(prior_corrections: list[str]) -> str:
    if not prior_corrections:
        return ""
    bullets = "\n".join(f"- {c}" for c in prior_corrections)
    return f"\n\nCorrections from past similar requests (do not repeat these mistakes):\n{bullets}"


def render_rules(rules: list[str]) -> str:
    """Standing rules (core/memory.py Memory.text, via MemoryStore.active_rules) —
    consolidated, deduplicated user preferences. Rendered before recent corrections:
    these are binding regardless of how similar the current request looks to any one
    past episode."""
    if not rules:
        return ""
    bullets = "\n".join(f"- {r}" for r in rules)
    return f"\n\nStanding rules from this user (binding, always apply):\n{bullets}"


DISTILLER_SYSTEM_PROMPT = """\
You turn a single user correction into a general, reusable rule for an agent that \
executes HTTP calls and scheduled tasks.

Given the original request, the action the agent took, and the user's correction, \
write one generalized, imperative rule (e.g. "always send Accept: application/json \
when calling api.example.com", not "send the header this one time"). Generalize past \
the specific URL/instruction where the correction clearly implies a broader pattern \
(e.g. a header requirement for a host applies to that whole host), but do not \
generalize further than the correction actually supports.

You will be shown existing rules with their ids. If this correction is saying the \
same thing as an existing rule, set `matches_existing_id` to that rule's id and leave \
`rule_text` as your best restatement (it will only be used if there's no match). If \
this correction contradicts an existing rule (the user now wants the opposite), set \
`supersedes_id` to that rule's id instead. Set `capability` to the capability name \
this rule scopes to (e.g. "http_call") if the correction is clearly capability-specific, \
or leave it null if it's general. Never set both `matches_existing_id` and \
`supersedes_id`."""


def render_existing_rules(existing: list[tuple[str, str]]) -> str:
    """`existing` is a list of (id, text) pairs."""
    if not existing:
        return "\n\nExisting rules: none yet."
    bullets = "\n".join(f"- [{rule_id}] {text}" for rule_id, text in existing)
    return f"\n\nExisting rules:\n{bullets}"


SUMMARIZER_SYSTEM_PROMPT = """\
You maintain a running summary of an ongoing conversation between a user and an \
agent that searches the web, calls APIs, runs code, schedules tasks, and generates \
reports on the user's behalf.

Given the prior summary (if any) and the next batch of turns (request/answer pairs), \
produce an updated, concise summary of the whole conversation so far. Preserve \
concrete facts, decisions, and artifacts the user might refer back to later — \
filenames, URLs, numbers, what was generated or scheduled — not a vague gist. Drop \
conversational filler. Keep it under roughly 200 words; if it's already near that \
length, prioritize the most recent and most concrete details over older, vaguer ones."""


def render_turns_for_summary(turns: list[tuple[str, str]]) -> str:
    """`turns` are (request, answer) pairs, oldest first — the summarizer's own
    rendering of what to fold into the running summary (see llm/summarizer.py)."""
    return "\n".join(f"- User: {request}\n  Agent: {answer}" for request, answer in turns)
