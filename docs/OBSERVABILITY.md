# Observability (Langfuse)

Langfuse tracing is integrated and off by default. This doc is everything needed
to turn it on once a Langfuse URL is available, plus what the integration
actually captures and why it's built the way it is.

## What gets traced

Requests and background jobs are wrapped in top-level **traces** at the API /
scheduler boundary via `observability.trace(...)`. Nested units of work become
child **spans** via `observability.span(...)`, and LLM calls become child
**generations** via `observability.generation(...)`.

Related calls share the exact same **`trace_id = episode_id`** (and `session_id = episode_id`).
Because Langfuse traces are keyed by `trace_id`, this unifies the initial execution,
any confirmation, and subsequent feedback into **one single Langfuse trace** in the dashboard:

```
[Trace: episode_id]
  ├─ api.chat                          (POST /chat and /chat/stream both use this trace)
  │    └─ agent.run
  │         input: the request text
  │         output: episode id, status, step count, capabilities per step, answer, reward
  │         ├─ agent.step              (one per step in the loop, index=0,1,2,...)
  │         │    ├─ {provider}.plan    (generation, only when planner != mock)
  │         │    │    input: system prompt + rendered capabilities/request/history/rules/corrections
  │         │    │    output: the candidates the model proposed for this step
  │         │    │    usage_details: real input/output token counts
  │         │    └─ capability.execute (only if the step didn't pause for confirmation)
  │         │         input: the chosen candidate's params
  │         │         metadata: capability name
  │         │         output: outcome.ok / status / error
  │         └─ agent.step              (repeats until an `answer` step or max_steps)
  │
  ├─ api.confirm                       (same trace_id = episode_id; /confirm and /confirm/.../stream)
  │    └─ agent.confirm
  │         input: episode id
  │         output: episode id, status, step count, capabilities per step, answer, reward
  │         ├─ capability.execute      (the previously-pending step, executed directly —
  │         │                           not inside its own agent.step, see below)
  │         └─ agent.step              (0+ more steps if the loop continues past the confirmed one)
  │
  └─ api.feedback                      (same trace_id = episode_id)
       └─ agent.feedback
            input: episode id
            metadata: score, has_correction
            └─ {provider}.distill      (generation, only when a correction was given)
                 input: the original request + every step's action taken + the correction
                 output: the distilled rule + match/supersede decision

[Trace: scheduler_trace_id]             (independent trace, source="scheduler")
  └─ scheduler.run
       └─ agent.run
            input: scheduled instruction
            output: episode id, status, step count, capabilities per step, answer, reward
            └─ agent.step              (one per step, each with a {provider}.plan generation
                                         and a capability.execute)
```

**Why the LLM calls are traced by hand instead of auto-instrumented:** the
obvious approach for the Claude case specifically is
`opentelemetry-instrumentation-anthropic`, which patches `AsyncMessages.create`
(the plain, non-beta messages resource). `LLMPlanner`/`LLMDistiller` are built
on PydanticAI's `Agent` + a provider `Model` (see docs/ARCHITECTURE.md "LLM
calls"); for Claude that's `AnthropicModel`, which internally calls
`client.beta.messages.create(...)` — a different resource class entirely, so
that instrumentor still never sees it. (Before the PydanticAI migration, this
project called `client.messages.parse()` directly, which posts to `/v1/messages`
without going through `.create()` at all — same conclusion, different reason.
Both were verified by reading the installed SDK's source, not assumed.) No
equivalent OTEL auto-instrumentation package is used for OpenAI or Google
either — manual instrumentation already covers all three uniformly through the
same `observability.generation()` call, so there's no reason to maintain a
different tracing mechanism per provider. The two call sites (`llm/llm_planner.py`,
`llm/distiller.py`) are wrapped by hand instead — see `core/observability.py`.

**Capability execution has its own span** — `core/agent.py:_execute()` wraps
every `Capability.execute()` call in a `capability.execute` span (`input` is
the candidate's params, `metadata.capability` is its name, `output` is the
outcome's ok/status/error). This was added specifically so `run_code`
(`capabilities/run_code.py`) has an audit trail of what code actually ran,
its exit status, and whether it ran in the Docker sandbox or the weaker
subprocess fallback (`outcome.payload["sandbox"]`) — worth checking here
first if a `run_code` step behaves unexpectedly. It's the single call site
used by both `Agent._advance`'s loop and `Agent.confirm`'s resumed step, so
it's covered uniformly; note that the one step `agent.confirm` itself
executes (the previously-pending one) runs directly under `agent.confirm`,
not inside its own `agent.step` span — only steps planned by
`Agent._advance`'s loop get one of those.

**What's *not* separately traced (yet):** policy selection (`Policy.select`)
happens inside `agent.step` but doesn't get its own child span — easy to add
later (`observability.span("policy.select", ...)` around the relevant lines
in `core/agent.py`) if there's ever a concrete reason to want that
granularity.

## Setup

### 1. Install the extra

```bash
pip install -e ".[observability]"
# or, without editable install:
pip install langfuse
```

Nothing else in the app imports `langfuse` unless this is installed *and*
tracing is turned on (below) — every call in `core/observability.py` is a no-op
when the package is missing or disabled, so this step can be skipped entirely in
environments that don't need traces (CI, the simulator, local dev without a key).

### 2. Set the environment variables

Once you have the Langfuse project URL, its public key, and its secret key:

```bash
export AGENTIC_RL_OBSERVABILITY_ENABLED=true    # our own on/off switch
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=<the URL you'll provide>   # e.g. https://cloud.langfuse.com,
                                                     # or your self-hosted instance's URL
```

`AGENTIC_RL_OBSERVABILITY_ENABLED` is this project's own setting
(`Settings.observability_enabled` in `core/config.py`) — it's what gates whether
`create_app()` calls `observability.setup()` at all. `LANGFUSE_*` are read
directly by the Langfuse SDK itself; this project doesn't proxy them through its
own config, so any Langfuse env var documented upstream (sampling, flush
behavior, etc. — see below) works without any code change here.

For a self-hosted Langfuse instance, server version must be **≥ 3.63.0**.

### 3. Verify credentials

```bash
python -c "from langfuse import get_client; print(get_client().auth_check())"
```

`True` means the keys and URL are correct and the SDK can reach the server.

### 4. Run the app and check the Langfuse UI

```bash
uvicorn agentic_rl.api.app:create_app --factory
```

Send one chat message through the UI or `curl -X POST localhost:8000/chat -d '{"message":"..."}'`,
then open the Langfuse project — a trace named `api.chat` should appear within
a few seconds (with child span `agent.run`; traces batch-flush; see `LANGFUSE_FLUSH_INTERVAL`
below if you want them to show up faster during manual testing).

## Configuration reference

Everything below is read directly by the `langfuse` package from the
environment — nothing is duplicated into this project's `Settings`.

| Env var | Default | Notes |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | — | required |
| `LANGFUSE_SECRET_KEY` | — | required |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | set for self-hosted or a non-default region |
| `LANGFUSE_TRACING_ENABLED` | `True` | set to the **capitalized string** `"False"` to disable at the SDK level — see gotcha below |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | 0.0–1.0; lower this in production if trace volume gets expensive |
| `LANGFUSE_FLUSH_AT` | `512` | batch size before an auto-flush |
| `LANGFUSE_FLUSH_INTERVAL` | `5.0` | seconds between auto-flushes |
| `LANGFUSE_TRACING_ENVIRONMENT` | `default` | tag traces by environment — consider setting this to the app's `Mode` (`sim`/`dev`/`prod_strict`) |
| `LANGFUSE_DEBUG` | `False` | verbose SDK logging |

## Known gotchas

- **`LANGFUSE_TRACING_ENABLED` is case-sensitive.** The SDK checks for the
  literal string `"False"`; `"false"` is silently ignored and tracing stays on.
  This is a known upstream quirk, not specific to this integration — worth
  remembering if you ever want to disable tracing via env var without unsetting
  the keys.
- **The Anthropic calls made here aren't covered by generic Anthropic OTEL
  auto-instrumentation.** See "Why the LLM calls are traced by hand" above —
  this is why the integration doesn't use
  `opentelemetry-instrumentation-anthropic` at all, even though it's the first
  thing Langfuse's own Anthropic integration docs suggest.
- **The client is a process-wide singleton** (`langfuse.get_client()`). This app
  calls `observability.setup()` once, in `create_app()`; don't call it again
  per-request.

## Privacy

Trace `input`/`output` include real request text, planner rationale, and
distilled rule text — i.e. actual user input and (if `planner != mock`) real
prompts/completions from whichever provider is configured, sent to wherever
`LANGFUSE_BASE_URL` points. If
that data is sensitive, either self-host Langfuse, or use its `mask` parameter
(`Langfuse(mask=...)` — construct a client directly instead of `get_client()`
if you need this — see `core/observability.py:setup()`) to redact fields before
they leave the process.

## Testing

`tests/test_observability.py`, `tests/test_planner.py`, and
`tests/test_distiller.py` include tests that run against the **real** installed
`langfuse` package with `LANGFUSE_TRACING_ENABLED=False` (via `monkeypatch`) —
this validates the actual SDK call shapes (`start_as_current_observation`,
`update_current_span`, `update_current_generation`, ...) without any network
traffic. If you upgrade the `langfuse` package, running
`pytest tests/test_observability.py tests/test_planner.py tests/test_distiller.py`
is the fastest way to catch an API-shape change.
