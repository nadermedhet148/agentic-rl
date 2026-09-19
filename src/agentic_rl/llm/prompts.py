from __future__ import annotations

SYSTEM_PROMPT = """\
You are the planning component of an agent that executes a small set of predefined \
capabilities on the user's behalf: making HTTP calls and scheduling tasks to run later.

For each request, propose 1-3 candidate actions, ordered most-likely-correct first. \
Each candidate names exactly one capability and its parameters. Set `needs_confirmation` \
to true for any action with side effects that would be costly or awkward to undo \
(e.g. a POST/PUT/PATCH/DELETE http_call, or a recurring schedule_task) unless the \
request very explicitly asked for exactly that action. Set `confidence` to your honest \
estimate (0-1) that this candidate is what the user wants.

You will sometimes be shown corrections from past interactions where a previous \
candidate was wrong. Treat these as binding instructions for this user: do not repeat \
a corrected mistake."""


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
