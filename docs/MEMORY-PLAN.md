# Memory — plan

## Context

v1 ([PLAN.md](PLAN.md)) has three kinds of memory, each with a gap:

| Tier | Today | Gap |
|---|---|---|
| **Procedural** (what works) | `LinUCBPolicy` `A`/`b` matrices in RAM | **Lost on restart.** The bandit re-learns from zero every boot. |
| **Semantic** (what the user wants) | nothing — corrections are raw strings on episodes | No consolidation (5 identical corrections = 5 prompt bullets), no contradiction handling, no way to state a preference without first making a mistake. |
| **Episodic** (what happened) | `episodes` + FTS5 `OR` of every token | `"fetch orders"` matches anything containing `fetch`; a correction about `api.example.com/orders` never surfaces for a differently-worded request to the same host. |

Scope agreed with the user: **persist policy state**, **semantic rules with LLM distillation + consolidation**, **better lexical retrieval** (no embeddings, no new external dependency beyond the Claude call already in use). Conversation sessions were out of scope for *this* pass — see "Session memory (v2)" below for the follow-up that added them. User-stated facts distinct from a conversation's own content (e.g. "I prefer metric units") remain out of scope.

## Design

```
                  ┌───────────────────────────────────────────────┐
 planner prompt ◄─│ Semantic: memories (rules)                    │◄─ consolidate ─┐
   "standing      │  text · capability? · support_count           │                │
    rules"        │  superseded_by · active · source_episode_ids  │                │
                  └───────────────────────────────────────────────┘                │
                  ┌───────────────────────────────────────────────┐                │
 planner prompt ◄─│ Episodic: episodes + ranked FTS5 retrieval     │── correction ──┘
   "recent        │  stopwords · min token overlap · host boost   │   (record_feedback)
    corrections"  └───────────────────────────────────────────────┘
                  ┌───────────────────────────────────────────────┐
 policy ◄────────►│ Procedural: policy_state table                 │  saved after every update,
                  │  policy_id → JSON state                        │  loaded in create_app
                  └───────────────────────────────────────────────┘
```

**Consolidation** is the new mechanism. On a correction, a `Distiller` turns it into a generalized rule and says whether it *matches* an existing rule (→ bump `support_count`) or *supersedes* one (→ mark old rule `superseded_by`, insert new). Active rules are injected into every plan — they're the user's standing instructions — instead of being fished out by keyword match. Episodic retrieval stays as a second, weaker channel for recent similar mistakes.

## Changes

### 1. Persist policy state — `policy/`, `core/store.py`, `core/agent.py`, `api/app.py`

- `policy/base.py`: add abstract `state_dict() -> dict` and `load_state(state: dict) -> None`.
  - `linucb.py`: `{"dim", "alpha", "arms": {arm_id: {"A": [[...]], "b": [...]}}}` (32×32 floats per arm as JSON lists — fine at this scale).
  - `epsilon.py`: `{"epsilon", "counts", "means"}`. `greedy.py`: `{}`.
- `core/store.py`: table `policy_state(policy_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL)`; `save_policy_state(policy_id, state)`, `load_policy_state(policy_id) -> dict | None`.
- `core/agent.py`: the two places that call `self._policy.update` (`_finalize_execution`, `record_feedback`) call `self._store.save_policy_state(self._policy.id, self._policy.state_dict())` afterwards. Keep it synchronous and unconditional — updates are rare (one per episode) and the write is a single row.
- `api/app.py` `create_app`: after `_build_policy`, `state = store.load_policy_state(policy.id); if state: policy.load_state(state)`.
- `sim/run.py`: no change (in-memory store, fresh policy per run is the point).

### 2. Better episodic retrieval — `core/store.py`

Rewrite `_fts_query` / `search_corrections`:

- **Stopwords**: drop `fetch get call please the a an from to for at with and on of run` (and any token < 2 chars) before building the query. Keep a short module-level set; it's tuned for request phrasing, not prose.
- **Host tokens**: extract URL hosts from the request (`re` on `https?://([^/\s]+)`) and add them as extra tokens, both to the FTS row at save time (new `hosts` column in `episodes_fts`) and to the query. Weight with `bm25(episodes_fts, 1.0, 0.5, 3.0)` so a shared host ranks above shared vocabulary.
- **Min overlap filter**: after FTS returns candidates, compute token-set overlap in Python and drop rows sharing < 2 non-stopword tokens (or < 1 if the query has only one). FTS5 can't express "at least N of these terms"; doing it post-query is simpler than building `AND`/`NEAR` combinations.
- `correction_count` keeps using `search_corrections` so `State.prior_correction_count` gets the same precision improvement for free.

Existing rows: `episodes_fts` is rebuilt from `episodes` on startup if the `hosts` column is missing (drop + recreate + reinsert). This is a dev project with a throwaway DB; a one-off rebuild is enough — no migration framework.

### 3. Semantic memory — new `core/memory.py`, new `llm/distiller.py`, `llm/prompts.py`, `llm/base.py`

**Model** (`core/models.py`):
```python
class Memory(BaseModel):
    id: str; created_at; updated_at
    kind: Literal["rule"] = "rule"          # room for "fact"/"preference" later
    text: str                                # generalized, imperative: "always send Accept: application/json to api.example.com"
    capability: str | None = None            # scope hint; None = applies to everything
    support_count: int = 1
    source_episode_ids: list[str]
    superseded_by: str | None = None
    active: bool = True
```

**Store** (`core/memory.py` → `MemoryStore`): takes the *same* `sqlite3` connection as `EpisodeStore` (add an `EpisodeStore.connection` property) so `:memory:` databases in tests and the sim stay a single DB. Tables `memories` + `memories_fts(memory_id, text)`. Methods: `add`, `get`, `bump_support(id, episode_id)`, `supersede(old_id, new_id)`, `deactivate(id)`, `active_rules(limit=50)` (ordered by `support_count` desc, then recency), `search(text, limit=5)` (reuses the ranked FTS query builder from §2).

**Distiller** (`llm/distiller.py`):
```python
class DistillResult(BaseModel):
    rule_text: str
    capability: str | None
    matches_existing_id: str | None   # equivalent to this rule → bump support
    supersedes_id: str | None         # contradicts this rule → replace it

class Distiller(ABC):
    id: str
    async def distill(self, correction: str, episode: Episode, existing: list[Memory]) -> DistillResult
```
- `ClaudeDistiller`: one `messages.parse()` call (same pattern as `llm/claude.py`), `claude-opus-5`, adaptive thinking, `effort: "low"`. Prompt: the request, the chosen action's capability + params, the correction, and the `existing` rules with their ids; asks for one generalized imperative rule and the match/supersede decision. Corrections are rare, so cost is negligible.
- `MockDistiller`: rule = correction text; `matches_existing_id` = first existing rule with the same normalized (lowercased, whitespace-collapsed) text; never supersedes. Used by tests and the sim.

**Consolidator** (`core/memory.py` → `Consolidator.consolidate(correction, episode) -> Memory`):
1. `existing = memory_store.search(correction, limit=5)` + any active rules scoped to the episode's capability (bounded).
2. `result = await distiller.distill(...)`.
3. If `matches_existing_id` (and it's active): `bump_support`, return it. Elif `supersedes_id`: `add` new, `supersede(old, new)`. Else `add`.
4. Guard: if the distiller names an id that isn't in `existing`, ignore the id and treat as new (LLMs invent ids).

**Wiring** (`core/agent.py`):
- Constructor gains `memory: MemoryStore` and `consolidator: Consolidator`.
- `record_feedback` becomes `async`; after `apply_feedback`, if `feedback.correction`: `await self._consolidator.consolidate(feedback.correction, episode)`. Route in `api/routes.py` becomes `async def feedback`.
- `run`: `rules = [m.text for m in self._memory.active_rules()]`; pass to the planner.

**Planner contract** (`llm/base.py`): `plan(state, tool_schemas, prior_corrections, rules: list[str] | None = None)`. `prompts.py` gains `render_rules()` — a "Standing rules from this user (binding)" section rendered *before* "Recent corrections on similar requests". `MockPlanner` ignores both (unchanged behavior); `ClaudePlanner` renders both.

### 4. API + UI — `api/routes.py`, `api/static/index.html`

- `GET /memories` (active rules, with `support_count`), `POST /memories {text, capability?}` (user adds a standing rule directly — same `MemoryStore.add`, no distillation), `DELETE /memories/{id}` (deactivate; a deactivated rule is never injected again).
- `index.html`: a **Rules** panel under Reward: text, "seen ×N", remove button; an input to add one. Refreshed with the other panels.

### File summary

| File | Change |
|---|---|
| `policy/base.py`, `linucb.py`, `epsilon.py`, `greedy.py` | `state_dict` / `load_state` |
| `core/store.py` | `policy_state` table; `connection` property; ranked retrieval (§2); FTS rebuild on missing `hosts` column |
| `core/models.py` | `Memory` |
| `core/memory.py` *(new)* | `MemoryStore`, `Consolidator` |
| `llm/distiller.py` *(new)* | `Distiller`, `DistillResult`, `ClaudeDistiller`, `MockDistiller` |
| `llm/base.py`, `mock.py`, `claude.py`, `prompts.py` | `rules` param; `render_rules` |
| `core/agent.py` | persist policy after update; inject rules; consolidate on correction; async `record_feedback` |
| `api/app.py`, `routes.py`, `static/index.html` | load policy state; build memory/consolidator/distiller; memory endpoints; Rules panel |
| `sim/run.py` | construct `MemoryStore` + `Consolidator(MockDistiller())`; `await agent.record_feedback` |

## Milestones (each ends green)

1. **Policy persistence** — state round-trip tests per policy; agent test: run episodes with a `tmp_path` DB, rebuild the app, assert `LinUCBPolicy.state_dict()` matches and the next `select` agrees.
2. **Retrieval ranking** — tests: `"fetch users from api.example.com"` retrieves a correction from `"fetch orders from api.example.com"`; `"fetch the weather"` does **not** retrieve it; hosts survive the FTS rebuild.
3. **MemoryStore + MockDistiller + Consolidator** — add / bump / supersede / deactivate / `active_rules` ordering; consolidating the same correction twice yields one rule with `support_count == 2`; invented ids are ignored.
4. **Agent wiring** — a correction via `record_feedback` produces a rule; the next `run` passes it to the planner (assert with a recording planner); sim still shows LinUCB learning and now accumulates rules.
5. **ClaudeDistiller** — unit test with a fake client (as in `tests/test_planner.py`); error chain wraps `RateLimitError` etc.
6. **API + UI** — endpoint tests; manual check of the Rules panel.

## Verification

- `.venv/Scripts/python -m pytest -q` — all green, no network.
- `.venv/Scripts/python -m agentic_rl.sim.run --policy linucb --episodes 300` — reward curve unchanged (still climbs to +1.00); print `len(memory.active_rules())` at the end and confirm it's 3 (one rule per scripted preference, each with a high `support_count`), not hundreds.
- Restart check: run the server with a file DB, chat + give 👍/👎 a few times, stop, start, `GET /metrics` history is intact **and** `policy_state` row exists (`sqlite3 agentic_rl.db "select policy_id, length(state) from policy_state"`).
- Manual (with `ANTHROPIC_API_KEY`): chat `"POST to https://httpbin.org/post with {a:1}"` → confirm → ✏️ "always include Content-Type: application/json" → `GET /memories` shows one generalized rule → next chat `"POST {b:2} to https://httpbin.org/anything"` proposes the header. Give the same correction again → still one rule, `support_count` 2.

## Session memory (v2)

The gap this closes: every `/chat` call started a brand-new, independent `Episode`
with no awareness of prior messages — asking for "a report about Langfuse" then, in
a separate message, "I want that as a PDF" failed, because nothing carried the word
"Langfuse" into the second episode's prompt. Unlike the semantic-rules tier above
(which captures *corrections* — standing behavioral preferences), this is about a
*conversation's own content* — a fourth kind of memory, deliberately kept separate:

| Tier | What it captures | Lifetime |
|---|---|---|
| Procedural / Semantic / Episodic | see table above | permanent, cross-conversation |
| **Conversation** (new) | this conversation's own turns — what was asked, what was answered, what was generated | scoped to one explicitly started/ended `Session`, not permanent |

**Explicit, not always-on** — the user chose a Start/End session control (UI:
`index.html`'s session panel above the chat log) over an always-on notion, so a
one-off question doesn't carry irrelevant context into an unrelated later one.

**Design** — `core/session.py:SessionStore` (`sessions` table: `status`,
`turn_count`, `summary`, `summarized_through`), same shared-connection pattern as
`MemoryStore`. `Episode` gained `session_id`. While a session is active,
`core/agent.py:Agent._session_context` builds `(summary, recent_turns)` from the
turns not yet folded into the summary (`EpisodeStore.list_session_episodes`) and
threads it through every `Planner.plan()` call as `conversation_summary`/
`conversation_turns`, rendered by `llm/prompts.py:render_conversation()` — same
shape as `render_rules`/`render_corrections`, new domain.

**Summarization, not unbounded growth** — every `settings.session_summarize_every`
turns (default 5), `Agent._finish_episode` folds the not-yet-summarized turns into
a rolling summary via a new `Summarizer` (`llm/summarizer.py:LLMSummarizer`/
`MockSummarizer` — same `Distiller`-shaped ABC, own system prompt instructing it to
preserve concrete facts/artifacts, not vague gist, since that's exactly what a
"make that a PDF" follow-up needs). A summarizer failure never fails the user's
response — see the `except Exception` in `_finish_episode`, tested in
`tests/test_agent_loop.py:test_summarizer_failure_does_not_break_the_turn`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component/data-model diagrams and
[CAPABILITIES.md](CAPABILITIES.md)/[REINFORCEMENT-LEARNING.md](REINFORCEMENT-LEARNING.md)
for how this composes with everything else — a session changes what the planner
sees, not how a step executes or how the bandit learns.
