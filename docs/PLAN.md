# agentic-rl — plan

## Context

Greenfield project (repo has only a README stub). Goal: an agent that executes predefined
capabilities (`http_call`, `schedule_task`) on the user's behalf and **learns from user
feedback via RL** so it stops repeating mistakes.

Decisions made with the user:
- **RL scope (v1):** RL over the agent's *decisions*, not over LLM weights. The LLM proposes
  candidate actions; a contextual-bandit policy picks among them; user feedback is the reward.
  Episode log is kept fine-tune-ready so a later phase can do DPO/GRPO on an open model.
- **Stack:** Python. FastAPI + SQLite + APScheduler + official `anthropic` SDK.
- **LLM:** pluggable behind a `Planner` protocol; ship Claude (`claude-opus-5`), add Ollama later.
- **Surface:** HTTP API + minimal static web UI (chat, feedback, tasks, episodes, reward curve).
- **Capabilities in v1:** only `http_call` and `schedule_task`.

Guiding constraints:
- Exploration never touches side-effecting actions in production. It happens in the simulator
  and on read-only/reversible actions only; production is near-greedy with a confirm gate.
- Explicit feedback outweighs implicit signals (avoid "user didn't complain" = reward hacking).
- The learning loop must be testable **without an LLM and without real HTTP** (mock planner,
  mock transport, scripted user) so learning curves can be measured in seconds.

## Architecture

```
User / Web UI ──► FastAPI ──► Agent loop ──► Capability registry
                    ▲             │             ├─ http_call      (tier: read | write)
                    │             │             └─ schedule_task  (tier: write; deferred agent run)
                    │             ├─ Planner (Claude | Mock | Ollama)  → N candidate actions
                    │             ├─ Policy  (LinUCB | epsilon-greedy | greedy) → pick one
                    │             ├─ Safety gate (confirm on write tier / low confidence)
                    │             ├─ Executor → outcome
                    │             └─ Reward collector → Episode store (SQLite) → Policy.update
                    │
              Scheduler (APScheduler, SQLite jobstore) ──► Agent loop (source=scheduler)
```

Agent loop per request — **multi-step**: an episode is a sequence of steps
(plan → select → confirm-gate → execute → observe), repeated until a step executes
the terminal `answer` capability or `settings.max_steps` is reached:
1. Build **state** once per episode: intent category (from the first step's planner
   call), source (user/scheduler), hour-of-day, rolling per-capability success rate
   (now tracked per step, not per episode), count of prior corrections for this intent.
2. Retrieve top-k **similar past episodes with corrections** (SQLite FTS5 on request text)
   and inject them into the planner prompt ("previously corrected: …"), alongside a
   summary of this episode's own steps so far (capability, params, outcome) once step > 0.
3. Planner returns **candidates** for the *next* step: `[{capability, params, rationale,
   confidence, needs_confirmation}]` via structured output (Pydantic schema). Once nothing
   further is needed, it proposes the `answer` capability with the final reply text.
4. `Policy.select(state_features, candidates)` → chosen candidate + `explored: bool`.
5. Safety gate: if chosen capability tier is `write` and (`needs_confirmation` or policy
   confidence low or mode=prod-strict) → episode status `pending_confirmation`, loop
   pauses; user confirms via API and the loop resumes from that step.
6. Execute → `Outcome{ok, status, payload, error}`.
7. Compute **implicit reward** for the step, persist the `Episode` (reward provisional),
   and — if the step wasn't `answer` and steps remain — go back to 2 for the next step.
   Streamed live to the UI as SSE `step` events (`POST /chat/stream`,
   `POST /confirm/{id}/stream`); `POST /chat`/`POST /confirm/{id}` remain available as
   plain blocking calls that just return the finished (or paused) `Episode`.
8. Once `answer` executes (or `max_steps` is hit), the episode's `implicit_reward` is the
   mean of its steps' implicit rewards. Later `POST /feedback {episode_id, score,
   correction?}` sets the episode's `final_reward` and updates **every step's** arm with
   that same reward (feedback is episode-level, not per step).

## Project layout

```
pyproject.toml                     (uv; deps: anthropic, fastapi, uvicorn, httpx, apscheduler,
                                    sqlalchemy, pydantic, numpy; dev: pytest, respx, ruff)
src/agentic_rl/
  core/
    models.py        Candidate, Action, Outcome, Episode, Feedback, State (pydantic)
    agent.py         Agent.run(request, source) — the loop above
    store.py         SQLAlchemy models + FTS5 episode search + reward finalization
    config.py        settings (mode: sim|dev|prod-strict, model id, db path)
  capabilities/
    base.py          Capability protocol: name, description, input_schema, tier, execute()
    registry.py      register/lookup; exports JSON schemas for the planner
    http_call.py     GET/POST/PUT/DELETE via httpx; tier = read for GET/HEAD else write
    schedule_task.py cron|run_at + instruction text → APScheduler job that calls Agent.run
  llm/
    base.py          Planner protocol: plan(request, state, tools_schema, prior_corrections) -> list[Candidate]
    claude.py        anthropic SDK, claude-opus-5, thinking adaptive, messages.parse() → Candidates
    mock.py          deterministic candidate sets from a fixture; used by tests + simulator
    prompts.py       system prompt; corrections rendered as few-shot "don't do X, do Y"
  policy/
    base.py          Policy protocol: select(x, candidates) -> (idx, explored); update(x, arm, reward)
    features.py      State + Candidate → feature vector (one-hot intent, capability, param-template
                     hash bucket, needs_confirmation, hour bucket, success-rate stats)
    linucb.py        LinUCB (per-arm A, b; alpha configurable; exploration off for write tier in prod)
    epsilon.py       epsilon-greedy over arm means (baseline)
    greedy.py        always pick highest planner confidence (control group)
  rl/
    reward.py        explicit: +1 / −1 / correction −1; implicit: exec ok +0.2, exec fail −0.5,
                     re-issued same intent within 10 min −0.3, scheduled task cancelled −0.5.
                     final = explicit if present else implicit; explicit weighted ×3 in update.
    export.py        dump episodes as JSONL (prompt, chosen, rejected, reward) for future fine-tuning
  scheduler/
    scheduler.py     APScheduler BackgroundScheduler + SQLAlchemyJobStore on the same SQLite file
  api/
    app.py           FastAPI app factory; mounts static/
    routes.py        POST /chat, POST /confirm/{episode_id}, POST /feedback, GET /tasks,
                     DELETE /tasks/{id}, GET /episodes, GET /metrics (rolling reward)
    static/index.html  vanilla JS: chat pane w/ 👍👎✏️ per reply, tasks table, episodes table,
                       rolling-reward chart (Chart.js from cdnjs)
  sim/
    env.py           httpx.MockTransport routes (JSON api, flaky endpoint, auth-required endpoint), fake clock
    user.py          ScriptedUser with hidden preferences (e.g. "always send Accept: application/json",
                     "never POST /orders without confirmation", "schedules in Europe/Berlin");
                     grades each action deterministically → explicit feedback + correction text
    run.py           CLI: --policy linucb|epsilon|greedy --episodes 500 → prints/plots reward curve
tests/
  test_capabilities.py, test_policy.py, test_reward.py, test_agent_loop.py, test_scheduler.py,
  test_learning.py (the sim: last-100 mean reward > first-100 mean, linucb > greedy)
```

## Milestones (each ends green)

1. **Scaffold** — `uv init`, layout above, config, empty FastAPI app, `pytest` passes trivially.
2. **Capabilities** — `Capability` protocol, registry, `http_call` (httpx, timeout, size cap),
   `schedule_task` (writes job; job body = deferred `Agent.run`). Unit tests with `respx`.
3. **Planner** — `Planner` protocol, `MockPlanner`, `ClaudePlanner` using
   `client.messages.parse()` with the `Candidates` Pydantic schema, `thinking={"type":"adaptive"}`,
   prompt caching on system prompt + tool schemas (stable prefix), corrections appended after.
   Consult the claude-api skill's `python/claude-api/README.md` + `tool-use.md` when writing this.
4. **Policy + reward + store** — features, LinUCB/epsilon/greedy, reward shaping, SQLite store
   with FTS5 search, `export.py`. Unit tests: LinUCB converges on a synthetic 3-arm problem.
5. **Agent loop + API** — `Agent.run`, confirm gate, `/chat` `/confirm` `/feedback` `/episodes`.
6. **Scheduler** — APScheduler wired to the same DB; scheduled runs flow through the loop and
   produce episodes; `/tasks` list/delete; cancel → negative implicit reward.
7. **Simulator + learning test** — `sim/` with `MockPlanner` + `MockTransport` + `ScriptedUser`;
   `test_learning.py` asserts learning; `sim/run.py` prints the curve.
8. **Web UI** — `index.html`; verify feedback round-trip and the reward chart moves.

## Key design details

- **Arm identity** for the bandit = `(capability, param_template_hash, needs_confirmation)`.
  `param_template_hash` hashes the *shape* of params (method, host, header keys, cron shape),
  not values, so learning generalizes across URLs.
- **Modes:** `sim` (full exploration), `dev` (explore on read tier only), `prod-strict`
  (greedy + always confirm write tier). Default `dev`.
- **Corrections are dual-use:** they set reward −1 for the chosen arm *and* are retrieved into
  future prompts, so the fix is visible on the very next similar request even before the
  bandit has converged.
- **Scheduled tasks** store the natural-language instruction, not a frozen action, so the
  policy can apply what it has learned by the time the job fires.
- **Episode schema** (store + export): request, source, a `steps` list (one entry per
  plan/select/execute pass: candidates considered, chosen idx, explored, outcome, per-step
  implicit reward), the terminal `answer` text (if reached), episode-level implicit_reward
  (mean of step rewards), explicit_score, correction, final_reward, planner id, policy id,
  timestamps. `rl/export.py` flattens this to one chosen-vs-rejected record per step, all
  sharing the episode's reward — this is the future DPO dataset.

## Verification

- `uv run pytest` — all unit tests plus `test_learning.py` (no network, no API key).
- `uv run python -m agentic_rl.sim.run --policy linucb --episodes 500` — mean reward of the
  last 100 episodes should clearly exceed the first 100; `--policy greedy` should be flat.
- Manual end-to-end (`ANTHROPIC_API_KEY` set, `uv run uvicorn agentic_rl.api.app:create_app --factory`):
  1. Chat: "fetch https://httpbin.org/json" → response + episode id; click 👍.
  2. Chat: "POST to https://httpbin.org/post with {a:1}" → `pending_confirmation`; confirm; 👎 with
     correction "always include Content-Type: application/json" → next similar request shows
     the header in the candidate and the correction in the episode's prompt.
  3. Chat: "every day at 9:00 fetch https://httpbin.org/uuid" → appears in `/tasks`; trigger it
     (short cron in dev) → new episode with `source=scheduler`.
  4. `/metrics` rolling reward reflects the feedback; chart in UI updates.
