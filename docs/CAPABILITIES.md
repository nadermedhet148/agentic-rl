# Capabilities (tools)

What each capability actually does, its safety tier, and how to add a new
one. For how a capability gets *chosen* (the RL side), see
[REINFORCEMENT-LEARNING.md](REINFORCEMENT-LEARNING.md); for how it fits into
the multi-step request loop, see [SEQUENCES.md](SEQUENCES.md).

## What a capability is

Every capability implements `Capability` (`capabilities/base.py`):

```python
class Capability(ABC):
    name: str
    description: str          # shown to the planner, verbatim
    input_schema: dict         # JSON schema, shown to the planner, verbatim

    def tier_for(self, params: dict) -> Tier: ...      # READ or WRITE, can depend on params
    async def execute(self, params: dict) -> Outcome: ...  # {ok, status, payload, error}
```

`CapabilityRegistry` (`capabilities/registry.py`) holds the set the agent
was built with and renders all of them into `tool_schemas()` — a plain list
of `{name, description, input_schema}` dicts that `LLMPlanner` writes
verbatim into its prompt (`llm/llm_planner.py`, `"Available capabilities:\n..."`).
The planner never calls a capability directly (no native tool-use loop) — it
only *proposes* `Candidate`s naming one; the bandit picks which candidate
runs, and only then does `Agent._execute` call `capability.execute(params)`.

## Safety tiers and the confirm gate

`Tier.READ` — no side effects, safe to auto-execute and safe for the bandit
to explore. `Tier.WRITE` — side-effecting or hard to reverse, gated:

| `Mode` | READ tier | WRITE tier |
|---|---|---|
| `sim` | auto-executes, explorable | auto-executes, explorable (simulator only — no real user to confirm with) |
| `dev` (default) | auto-executes | pauses for confirmation **only if** the planner set `needs_confirmation: true` on that candidate |
| `prod_strict` | auto-executes | **always** pauses for confirmation, regardless of the planner's flag |

An **unregistered** capability name (the planner hallucinated one, or it was
never wired up) is always treated as WRITE + forced confirmation
(`core/agent.py:_needs_confirmation`, `capability_known=False`) — confirming
it then fails cleanly with `"unknown capability: ..."` rather than the loop
silently doing nothing.

## The six capabilities

| Name | Tier | Typically confirmed? | One-line purpose |
|---|---|---|---|
| [`web_search`](#web_search) | read | never | DuckDuckGo search, no API key |
| [`http_call`](#http_call) | read (GET/HEAD/OPTIONS) / write (POST/PUT/PATCH/DELETE) | write methods, if the planner flags them | arbitrary HTTP request |
| [`schedule_task`](#schedule_task) | write | if the planner flags it | defer a natural-language instruction to a cron/one-off trigger |
| [`run_code`](#run_code) | write | **always** (prompted explicitly) | run a Python snippet in a sandbox |
| [`generate_report`](#generate_report) | write | if the planner flags it | render text as a PDF |
| [`answer`](#answer) | read | never | terminal step — the loop's own "I'm done" signal |

"Typically confirmed" describes `dev` mode (the default) — it's the
planner's judgment call per `llm/prompts.py:SYSTEM_PROMPT`, not a hard rule,
except where noted.

---

### `web_search`

`capabilities/web_search.py` · input: `{query: string, max_results?: int}`

Backed by `ddgs` (the `DuckDuckGo search` library) via a `SearchPort`
protocol — `DdgsSearchPort` is the real implementation, tests inject a fake.
`ddgs` is synchronous, so `execute()` runs it in a thread
(`asyncio.to_thread`) with a timeout (`AGENTIC_RL_WEB_SEARCH_TIMEOUT_S`,
default 10s) so it can't stall the agent loop. Returns
`payload.results: [{title, url, snippet}]`. No API key, no cost.

### `http_call`

`capabilities/http_call.py` · input: `{method, url, headers?, json_body?}`

A single shared `httpx.AsyncClient` (`api/app.py`) makes the request; tier
depends on `method` — `GET`/`HEAD`/`OPTIONS` are read, everything else
(`POST`/`PUT`/`PATCH`/`DELETE`) is write. Response body is truncated to
`AGENTIC_RL_HTTP_MAX_BODY_BYTES` (default 256KB) and JSON-decoded if
possible, else returned as text. This is the most general-purpose
capability — it's also the only one whose tier isn't fixed, since the same
capability can be perfectly safe (`GET`) or genuinely risky (`DELETE`)
depending on what the planner asks it to do.

### `schedule_task`

`capabilities/schedule_task.py` · input: `{instruction, cron? | run_at?, timezone?}`
(exactly one of `cron`/`run_at`)

Talks to `AgentScheduler` (APScheduler) through a narrow `SchedulerPort`
protocol to avoid an import cycle. **Stores the natural-language
`instruction`, not a frozen action** — when the job fires, it re-enters
`Agent.run(instruction, source="scheduler")` from scratch, so anything the
policy has learned (or any standing rule added) since the job was scheduled
applies at fire time, not scheduling time. Always write-tier (a recurring
job is a real ongoing commitment). Cancelling a scheduled job
(`DELETE /tasks/{job_id}`) penalizes the *episode that originally scheduled
it* (`rl/reward.py: CANCELLED`), not the policy directly.

### `run_code`

`capabilities/run_code.py` · input: `{code: string}`

Runs a Python snippet and returns its stdout. **Always write-tier, and the
system prompt explicitly instructs the planner to always set
`needs_confirmation: true` for it** (`llm/prompts.py`) — real code
execution on the server is categorically riskier than the other write-tier
capabilities, so it doesn't rely solely on the planner's own judgment the
way `http_call`/`schedule_task` do.

Two `CodeRunner` implementations, picked once at app startup
(`api/app.py`, via `docker_available()`):

- **`DockerCodeRunner`** (preferred) — `docker run --rm --network none
  --memory=<N>m --cpus=<N> <image> python script.py`. Network-isolated,
  resource-capped, discarded after one run. `docker_available()` checks not
  just `shutil.which("docker")` but that the daemon actually answers
  (`docker info`) — Docker Desktop can be *installed* but not *running*,
  which would otherwise make every call fail instead of falling back.
- **`SubprocessCodeRunner`** (fallback) — a plain `python -I script.py`
  subprocess of the app's own interpreter. `-I` (isolated mode) drops the
  user site-packages dir and `PYTHONPATH`/`PYTHONHOME` influence, but this
  is **not a security sandbox** — the script shares the app process's OS
  permissions (filesystem, network) and installed packages.

Either way: a fresh `tempfile.TemporaryDirectory()` per call (nothing
persists between calls), a hard timeout (`AGENTIC_RL_RUN_CODE_TIMEOUT_S`,
default 10s), and stdout/stderr truncated to
`AGENTIC_RL_RUN_CODE_MAX_OUTPUT_BYTES` (default 64KB).
`outcome.payload["sandbox"]` reports which path actually ran
(`"docker"`/`"subprocess"`) — check this first if a `run_code` step behaves
unexpectedly (also traced — see `capability.execute` in
[OBSERVABILITY.md](OBSERVABILITY.md)). Guidance to the planner: use it for
computation/data reshaping it can't do reliably itself (exact math,
sorting, JSON/CSV reshaping) — never to fetch a URL, that's what
`http_call`/`web_search` are for.

### `generate_report`

`capabilities/generate_report.py` · input: `{title, content, filename?}`

Renders `content` as a simple PDF via `fpdf2`: blank lines start a new
paragraph, a line starting `# ` is a bold heading, a line starting `- ` is a
bullet, anything else is wrapped body text. Saved under
`AGENTIC_RL_REPORTS_DIR` (default `reports/`) and served back at
`GET /reports/<filename>` (mounted in `api/app.py`, registered *before* the
catch-all `/` static mount so it isn't shadowed).

Two things worth knowing if you're touching this file:
- **`filename` is LLM-controlled input, sanitized against path traversal**
  (`_safe_filename`) — everything outside `[A-Za-z0-9_-]` is collapsed to a
  single `-`, so `../../evil.pdf` becomes `evil.pdf`, always confined to
  `reports_dir` (double-checked by resolving the final path and verifying
  it's still inside `reports_dir` before writing).
- **Text is sanitized to Latin-1** (`_latin1_safe`) before rendering —
  `fpdf2`'s core fonts (Helvetica/Times/Courier) only support Latin-1/
  WinAnsi, so emoji, smart quotes, or non-Latin text (plausible in an LLM
  answer or a `web_search` snippet fed into the report) get replaced rather
  than crashing the capability. A real Unicode font could be embedded later
  if this starts to matter more than "for now."

Write-tier (it persists a file), but — unlike `run_code` — there's no
system-prompt instruction forcing confirmation; it's treated like any other
write-tier action (planner's judgment in `dev` mode), since writing a report
file is a far more benign, trivially-reversible side effect than executing
arbitrary code.

### `answer`

`capabilities/answer.py` · input: `{text: string}`

Not a tool that touches anything external — it's how the multi-step loop
(`core/agent.py:_advance`) knows it's *done*. `Agent.__init__` auto-registers
it on every agent (`if registry.get_or_none("answer") is None: ...`), so it
doesn't need wiring at each construction site the way the other five do.
Always read-tier, and it **competes as an ordinary bandit arm** — nothing
special-cases it in the policy or feature code, so the bandit genuinely
learns *when stopping is the right call* the same way it learns anything
else. `execute()` just validates `text` is non-empty and echoes it back as
`payload.text`; `Agent._execute_step` is what actually reads
`step.action.candidate.capability == "answer"` and sets `episode.answer`.

---

## Adding a new capability

Follow the pattern the six above already establish:

1. **New file** in `capabilities/`, subclassing `Capability`. If it talks to
   something you don't want to hit for real in tests (a network call, a
   subprocess, an external SDK), define a narrow **port protocol** for it
   (see `SearchPort`, `SchedulerPort`, `CodeRunner`) and accept an instance
   of it in `__init__` — the real implementation goes in the same file, a
   fake goes in the test file.
2. **`tier_for()`** — `Tier.READ` if truly side-effect-free, `Tier.WRITE`
   otherwise (default to WRITE if unsure; the confirm gate is the safety
   net, not a formality). If genuinely risky (like `run_code`), add a line
   to `llm/prompts.py:SYSTEM_PROMPT` telling the planner to always set
   `needs_confirmation: true` for it — the tier alone only gates
   `prod_strict` mode automatically.
3. **`execute()`** — validate required params and return
   `Outcome(ok=False, error=...)` on bad input (don't raise; `Agent._execute`
   catches exceptions as a backstop, but a clear `error` string is much more
   useful to the planner on the next step than a stack trace). Truncate any
   unbounded output the same way `http_call`/`web_search`/`run_code` do.
4. **Register it** in `api/app.py`'s `create_app()`, and in
   `sim/run.py`/`tests/test_agent_loop.py`'s `make_agent` helpers if the
   simulator or agent-loop tests should exercise it.
5. **Settings**, if it needs any (a timeout, a size cap, a directory) — add
   to `core/config.py: Settings` following the existing naming convention
   (`<capability>_<thing>`), and document the env var in `.env.example`.
6. **Tests** in `tests/test_capabilities.py`: `tier_for()` behavior, a
   success case, a failure case (bad input, backend error), using the fake
   port if you defined one. See any of the existing `# --- <name> ---`
   sections for the shape.
7. **Docs** — add a row + subsection here, and to the capability lists in
   [PLAN.md](PLAN.md) and [ARCHITECTURE.md](ARCHITECTURE.md) (component
   diagram, package layout, runtime composition diagram).

That's it — nothing in `core/agent.py`, the policy, or the feature code
needs to know a new capability exists. It shows up in the planner's prompt
automatically via `CapabilityRegistry.tool_schemas()`, and the bandit starts
learning about it the first time it's proposed.
