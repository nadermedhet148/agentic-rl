# Manual testing guide

A worked example for every distinct case the system handles, as `curl` commands
(cross-platform, work in Git Bash / PowerShell / any shell) against a running
instance. Every example below was actually run against a live instance while
writing this doc — not just theoretically correct.

Start the app first (`run.bat` or `uvicorn agentic_rl.api.app:create_app --factory`),
then run these from another terminal. Replace `<episode_id>`/`<job_id>`/`<memory_id>`
with real ids from the response just before each step — they're randomly generated
per run.

**Cost note:** with `AGENTIC_RL_PLANNER` set to `claude`/`openai`/`google`, every
`/chat` call is a real, billed LLM request. Set `AGENTIC_RL_PLANNER=mock` in
`.env` for free, offline runs of everything except the "provider switching"
section — the mock planner's heuristic (`llm/mock.py:heuristic_default_fn`)
still exercises the same code paths, just without real reasoning.

## 1. Read-tier action — auto-executes, no confirmation

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"fetch https://httpbin.org/json"}'
```
Expect: `"status":"executed"`, `"outcome":{"ok":true,...}` — a GET is read-tier,
so it runs immediately (see `core/agent.py:_needs_confirmation`).

## 2. Write-tier action the model flags — needs confirmation, then confirm

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"delete the order at https://httpbin.org/delete"}'
# -> {"status":"pending_confirmation", "action":{"candidate":{"needs_confirmation":true,...}}, "id":"<episode_id>", ...}

curl -s -X POST http://127.0.0.1:8000/confirm/<episode_id>
# -> {"status":"executed", "outcome":{"ok":true,...}}
```
Confirming twice, or an unknown episode id, are also worth trying:
```bash
curl -s -X POST http://127.0.0.1:8000/confirm/<episode_id>   # second time -> 409
curl -s -X POST http://127.0.0.1:8000/confirm/does-not-exist # -> 404
```

## 3. Write-tier action the model *doesn't* flag — auto-executes anyway (in `dev` mode)

Confirmation is gated on the planner's own `needs_confirmation` flag in `dev`
mode, not on tier alone — only `prod_strict` mode forces confirmation on every
write-tier action unconditionally (`core/config.py: Mode`):

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"schedule a daily check of https://httpbin.org/json every day at 9am UTC"}'
```
A real run of this against Gemini returned `"needs_confirmation":false` and
`"status":"executed"` directly — the model judged a routine daily check safe
enough not to ask. Compare against #2 to see the model actually reasoning about
risk per-request, not just per-capability.

## 4. Feedback — thumbs up / thumbs down (no correction)

```bash
curl -s -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" \
  -d '{"episode_id":"<episode_id>","score":1}'
# final_reward becomes 1.0

curl -s -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" \
  -d '{"episode_id":"<episode_id>","score":-1}'
# final_reward becomes -1.0
```

## 5. Feedback with a correction — creates a memory rule

```bash
curl -s -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" \
  -d '{"episode_id":"<episode_id>","score":-1,"correction":"always include header Accept: application/json when calling httpbin.org"}'

curl -s http://127.0.0.1:8000/memories
```
A real run produced the generalized rule *"Always include the header 'Accept:
application/json' when making HTTP calls to httpbin.org."* with
`"capability":"http_call"` — the distiller both generalized the wording and
inferred the right scope on its own.

## 6. Repeat the same correction — bumps `support_count`, doesn't duplicate

Send the same (or a near-duplicate) correction again on a different episode:
```bash
curl -s -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" \
  -d '{"episode_id":"<another_episode_id>","score":-1,"correction":"always include header Accept: application/json when calling httpbin.org"}'

curl -s http://127.0.0.1:8000/memories
```
Expect the same rule's `"support_count"` to increment (2, 3, ...) rather than a
second near-identical rule appearing — the distiller is shown existing rules and
asked to match, not just append (`core/memory.py: Consolidator`).

## 7. Reissue the same request — penalizes the *earlier* episode

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"fetch https://httpbin.org/uuid"}'
# note the id and implicit_reward (e.g. 0.2)

curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"fetch https://httpbin.org/uuid"}'
# same message again

curl -s "http://127.0.0.1:8000/episodes?limit=5"
```
Expect the *first* episode's `implicit_reward` to have dropped by `0.3` (e.g.
`0.2` → `-0.1`) — re-asking the same thing shortly after implies the first
answer didn't satisfy the user (`rl/reward.py: REISSUED`, applied in
`core/store.py: maybe_penalize_reissue`).

## 8. Scheduled tasks — create, list, cancel

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"every day at 9am fetch https://httpbin.org/json"}'
# -> outcome.payload.job_id

curl -s http://127.0.0.1:8000/tasks
# -> [{"id":"<job_id>","instruction":"...","next_run_time":"..."}]

curl -s -X DELETE http://127.0.0.1:8000/tasks/<job_id>
# -> {"status":"cancelled","job_id":"<job_id>"}

curl -s http://127.0.0.1:8000/tasks   # -> []
```
Cancelling also penalizes the episode that scheduled it — check
`GET /episodes` for that episode's `implicit_reward` before and after.

## 9. Memory rules — add and remove directly (no correction needed)

```bash
curl -s -X POST http://127.0.0.1:8000/memories -H "Content-Type: application/json" \
  -d '{"text":"never delete production orders without manager approval","capability":"http_call"}'
# -> {"id":"<memory_id>", "support_count":1, ...}

curl -s -X DELETE http://127.0.0.1:8000/memories/<memory_id>
# -> {"status":"deactivated","id":"<memory_id>"}

curl -s -X DELETE http://127.0.0.1:8000/memories/does-not-exist   # -> 404
```

## 10. Unknown / unrecognized request — forced confirmation, clean failure

Ask for something the registered capabilities (`http_call`, `schedule_task`)
genuinely can't do:
```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"book me a flight to Tokyo"}'
```
Expect `"status":"pending_confirmation"`. What the model actually proposes here
is worth noting because it's more honest than "refuse" or "crash": a real
Gemini run against this exact message returned an `http_call` to a made-up
placeholder URL (`https://api.example.com/flights/book`), with `rationale`
*"...no specific flight booking API or service is configured or mentioned. I am
providing a placeholder HTTP call, but it is highly unlikely to succeed without
proper API details"* and `needs_confirmation: true` — the model flagged its own
uncertainty rather than pretending. (If a model instead proposes a capability
name that isn't registered at all, that's *also* forced to
`pending_confirmation` — see `core/agent.py:_needs_confirmation`,
`capability_known=False` — but that's not what happened in this run.)

Confirming it then fails safely rather than crashing:
```bash
curl -s -X POST http://127.0.0.1:8000/confirm/<episode_id>
# -> {"status":"executed", "outcome":{"ok":false,"error":"ConnectError: ..."}}
# (or "unknown capability: ..." if the model proposed an unregistered capability name)
```

## 11. Episodes, metrics, and rules — the read-only views

```bash
curl -s "http://127.0.0.1:8000/episodes?limit=10"
curl -s "http://127.0.0.1:8000/metrics?n=100"     # rolling reward + mean
curl -s "http://127.0.0.1:8000/memories?limit=50"
```
All three back the web UI's panels directly — open http://127.0.0.1:8000 to see
the same data rendered (chat log, reward chart, rules list, tasks table).

## 12. Switching provider (`claude` / `openai` / `google` / `mock`)

Provider is picked once at startup from `AGENTIC_RL_PLANNER` — changing it
needs a restart, not a request:
```bash
# .env: AGENTIC_RL_PLANNER=openai, AGENTIC_RL_LLM_MODEL=gpt-5.1, OPENAI_API_KEY=...
# then restart: run.bat
```
After restarting, repeat case #1 and check `"planner_id"` in the response —
it reflects whichever provider actually ran (`llm/llm_planner.py: LLMPlanner.id`).
See `.env.example` and `docs/ARCHITECTURE.md` "LLM calls" for what differs
between providers (credential resolution, effort control).

## 13. Switching mode (`sim` / `dev` / `prod_strict`)

Also startup-only. With `AGENTIC_RL_MODE=prod_strict`, repeat case #3 — the
same "the model didn't flag it" schedule_task request should now come back
`"status":"pending_confirmation"` regardless of what the model decided, since
`prod_strict` forces confirmation on every write-tier action unconditionally.

## 14. Observability (if `AGENTIC_RL_OBSERVABILITY_ENABLED=true`)

No separate test — every case above already produces a trace when this is on.
Check your Langfuse project: one `agent.run`/`agent.confirm`/`agent.feedback`
trace per request above, with a nested `{provider}.plan`/`{provider}.distill`
generation wherever a real LLM call happened. See `docs/OBSERVABILITY.md`.
