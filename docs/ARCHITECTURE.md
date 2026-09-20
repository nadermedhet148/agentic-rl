# Architecture

This document is the structural map of agentic-rl: what the pieces are, how they're
packaged, and how they're wired together at runtime. For *how requests flow through
the system step by step*, see [SEQUENCES.md](SEQUENCES.md). For the original design
rationale, see [PLAN.md](PLAN.md) and [MEMORY-PLAN.md](MEMORY-PLAN.md). For how the
bandit actually learns, see [REINFORCEMENT-LEARNING.md](REINFORCEMENT-LEARNING.md);
for what each capability does, see [CAPABILITIES.md](CAPABILITIES.md).

## System overview

A single Python process serves an HTTP API + static web UI, runs a background
scheduler thread, and reads/writes one SQLite file. There is no separate worker
process and no external database — the whole system fits in one `uvicorn` process
for local development and small deployments.

```mermaid
graph TB
    subgraph Client
        UI["Web UI<br/>(static/index.html)"]
    end

    subgraph Process["agentic-rl process"]
        API["FastAPI app<br/>(api/app.py, api/routes.py)"]
        Agent["Agent<br/>(core/agent.py)"]
        Sched["AgentScheduler<br/>(scheduler/scheduler.py)<br/>APScheduler background thread"]
    end

    subgraph External
        Claude["Claude API<br/>(via PydanticAI Agent + AnthropicModel)"]
        HTTP["Arbitrary HTTP endpoints<br/>(via http_call)"]
        DDG["DuckDuckGo<br/>(via web_search, ddgs)"]
        Docker["Docker<br/>(run_code sandbox, if installed & running —<br/>else a plain subprocess)"]
        Langfuse["Langfuse<br/>(traces, optional)"]
    end

    DB[("SQLite file<br/>episodes · episodes_fts<br/>policy_state · memories · memories_fts<br/>apscheduler_jobs")]
    Reports[("reports_dir<br/>generate_report's PDFs,<br/>served at /reports/*")]

    UI <-->|"REST + JSON"| API
    API --> Agent
    API --> Sched
    Sched -->|"fires job → Agent.run(source=scheduler)"| Agent
    Agent -->|reads/writes| DB
    Sched -->|reads/writes jobs| DB
    Agent -.->|plan / distill| Claude
    Agent -.->|execute http_call| HTTP
    Agent -.->|execute web_search| DDG
    Agent -.->|"execute run_code<br/>(--network none, --rm)"| Docker
    Agent -->|execute generate_report| Reports
    API -->|"GET /reports/*"| Reports
    Agent -.->|"spans + generations<br/>(off by default)"| Langfuse
```

Langfuse tracing is entirely optional and off by default (`AGENTIC_RL_OBSERVABILITY_ENABLED=false`) —
see [OBSERVABILITY.md](OBSERVABILITY.md) for what gets traced and how to turn it on.

## Component diagram

`Agent` is the hub — it owns no state itself beyond references to its
collaborators, and every request passes through the same object.

```mermaid
graph TB
    Agent["Agent<br/>(core/agent.py)"]

    Planner["Planner<br/>(llm/base.py)"]
    LLMPlanner["LLMPlanner<br/>(provider: claude|openai|google)"]
    MockPlanner["MockPlanner"]
    Planner --- LLMPlanner
    Planner --- MockPlanner

    Policy["Policy<br/>(policy/base.py)"]
    LinUCB["LinUCBPolicy"]
    Epsilon["EpsilonGreedyPolicy"]
    Greedy["GreedyPolicy"]
    Policy --- LinUCB
    Policy --- Epsilon
    Policy --- Greedy

    Registry["CapabilityRegistry<br/>(capabilities/registry.py)"]
    HttpCap["HttpCallCapability"]
    SchedCap["ScheduleTaskCapability"]
    SearchCap["WebSearchCapability"]
    AnswerCap["AnswerCapability<br/>(terminal step — Agent auto-registers it)"]
    RunCodeCap["RunCodeCapability<br/>(DockerCodeRunner | SubprocessCodeRunner)"]
    ReportCap["GenerateReportCapability"]
    Registry --- HttpCap
    Registry --- SchedCap
    Registry --- SearchCap
    Registry --- AnswerCap
    Registry --- RunCodeCap
    Registry --- ReportCap

    Store["EpisodeStore<br/>(core/store.py)"]
    Memory["MemoryStore<br/>(core/memory.py)"]
    Consolidator["Consolidator<br/>(core/memory.py)"]
    Distiller["Distiller<br/>(llm/distiller.py)"]
    LLMDistiller["LLMDistiller<br/>(provider: claude|openai|google)"]
    MockDistiller["MockDistiller"]
    Distiller --- LLMDistiller
    Distiller --- MockDistiller

    Sessions["SessionStore<br/>(core/session.py)"]
    Summarizer["Summarizer<br/>(llm/summarizer.py)"]
    LLMSummarizer["LLMSummarizer<br/>(provider: claude|openai|google)"]
    MockSummarizer["MockSummarizer"]
    Summarizer --- LLMSummarizer
    Summarizer --- MockSummarizer

    Agent --> Planner
    Agent --> Policy
    Agent --> Registry
    Agent --> Store
    Agent --> Memory
    Agent --> Consolidator
    Agent --> Sessions
    Agent --> Summarizer
    Consolidator --> Distiller
    Consolidator --> Memory

    SchedCap -.->|SchedulerPort protocol| AgentScheduler["AgentScheduler"]
    AgentScheduler -->|"fires: Agent.run(instruction, source='scheduler')"| Agent

    Store -.->|"shares sqlite3.Connection<br/>(EpisodeStore.connection)"| Memory
    Store -.->|"shares sqlite3.Connection<br/>(EpisodeStore.connection)"| Sessions
```

### LLM calls — model-agnostic via PydanticAI

`LLMPlanner` and `LLMDistiller` (`llm/llm_planner.py`, `llm/distiller.py`) are
built on [PydanticAI](https://ai.pydantic.dev)'s `Agent` + a provider `Model`
rather than calling any single vendor's SDK directly. **One `provider`
constructor argument** (`"claude"` / `"openai"` / `"google"`, driven end-to-end
by the single `AGENTIC_RL_PLANNER` setting — see `api/app.py`'s
`_build_planner`/`_build_distiller`) selects the model; everything else —
prompt construction, structured-output validation, error mapping, tracing — is
identical regardless of which one is picked. All provider-specific knowledge
lives in one place, `llm/providers.py`:

- **`build_model(provider, model_name) -> Model`** constructs the right
  PydanticAI `Model` (`AnthropicModel` / `OpenAIChatModel` / `GoogleModel`).
- **`build_settings(provider, *, max_tokens, effort) -> ModelSettings`**
  constructs the matching settings. PydanticAI has one *unified* field for
  reasoning depth, `ModelSettings.thinking`, but it doesn't carry full
  granularity for every provider — verified by reading the installed SDK, not
  assumed:
  - **Claude**: any truthy `thinking` value just switches on adaptive
    thinking for adaptive-capable models (the specific effort level is
    dropped in that branch), so Claude additionally gets the provider-specific
    `anthropic_effort` field, which does carry the level.
  - **OpenAI**: the unified `thinking` value maps one-for-one onto OpenAI's own
    reasoning-effort scale (same literal names), so no provider-specific
    override is needed.
  - **Google**: gets `thinking=True` only — no comparable effort-level scale
    is exposed here.
- **Structured output** (`Agent(output_type=_PlanResponse)` /
  `Agent(output_type=DistillResult)`) drives each provider's own native
  schema-validated-response mechanism.
- **Errors** surface as PydanticAI's own `ModelHTTPError` (`.status_code`
  distinguishes 404/429/other) and `ModelAPIError` (connection failures) —
  uniform across providers — rather than each vendor's own exception
  hierarchy; both `LLMPlanner`/`LLMDistiller` map them to the same
  `RuntimeError` messages ("model not found" / "rate limited" / "api error" /
  "connection error") regardless of which provider raised them.
- **Credential resolution differs by provider, verified rather than assumed
  uniform**: `anthropic.AsyncAnthropic()` resolves its full credential chain
  (API key, `ant auth login` profiles, ...) lazily, so `build_model` for
  `"claude"` constructs one itself and hands it to
  `AnthropicProvider(anthropic_client=...)` to preserve that. `openai.AsyncOpenAI()`
  and `google.genai.Client()` both raise immediately if no key is available —
  there's no fuller chain to preserve underneath them — so `"openai"`/`"google"`
  are left to PydanticAI's own provider inference, which already produces a
  clear, actionable `UserError`. **Concretely: constructing an
  `LLMPlanner`/`LLMDistiller` for `openai`/`google` can raise at app startup**
  if the corresponding key isn't set; for `claude` it can't (it fails lazily,
  at request time, on whatever request first needs it).
- **A model-id/provider mismatch guard**: `build_model` rejects a `claude-*`
  model name paired with `provider="openai"` (and the other cross-provider
  combinations) with a clear error, rather than letting a forgotten
  `AGENTIC_RL_LLM_MODEL` update surface as a confusing remote 404.
- **Testing** injects a PydanticAI `Model` test double (`TestModel` for
  "just give me this structured output back", `FunctionModel` when a test needs
  to inspect the outgoing prompt or simulate a specific `ModelHTTPError`) via
  each class's `pydantic_model=` constructor argument — not a fake provider
  client, because each real `Model` calls its own SDK's request method
  internally (e.g. `AnthropicModel` calls `client.beta.messages.create(...)`)
  and a hand-rolled fake can't track that reliably across SDK versions.

**Why `Agent` depends on this many collaborators instead of fewer:** each one is a
separately swappable strategy — `Planner`, `Distiller`, and `Summarizer` all swap for
`Mock*` implementations with no network in tests and the simulator; `Policy` swaps
between three baselines to prove the RL policy actually beats "no learning";
`EpisodeStore`/`MemoryStore`/`SessionStore` are the three halves of memory
(episodic/semantic/conversation — see [MEMORY-PLAN.md](MEMORY-PLAN.md)). None of
them depend on `Agent`, so the dependency graph is a strict DAG and every
collaborator is unit-testable alone.

## Package layout

```mermaid
graph LR
    subgraph src/agentic_rl
        core["core/<br/>agent, models, store,<br/>memory, session, text, config"]
        capabilities["capabilities/<br/>base, registry,<br/>http_call, schedule_task,<br/>web_search, answer,<br/>run_code, generate_report"]
        llm["llm/<br/>base, llm_planner, mock,<br/>distiller, summarizer,<br/>prompts, providers"]
        policy["policy/<br/>base, features,<br/>linucb, epsilon, greedy"]
        rl["rl/<br/>reward, export"]
        scheduler["scheduler/<br/>scheduler.py"]
        api["api/<br/>app, routes, static/"]
        sim["sim/<br/>env, user, run"]
    end

    api --> core
    api --> capabilities
    api --> llm
    api --> policy
    api --> scheduler
    core --> capabilities
    core --> llm
    core --> policy
    core --> rl
    capabilities --> scheduler
    sim --> core
    sim --> capabilities
    sim --> llm
    sim --> policy
```

`core` is the only package every other package depends on (directly or through
`capabilities`); it has no dependency on `api`, `llm`, `policy`, or `scheduler`
implementations — only on the `Planner`/`Policy`/`Distiller`/`Capability` *protocols*,
which live in each package's own `base.py`. This is what keeps `Mock*`
substitution possible without touching `core`.

## Data model

Everything lives in one SQLite file, opened once by `EpisodeStore` and shared (via
`EpisodeStore.connection`) with `MemoryStore`, `SessionStore`, and APScheduler's own
job store.

```mermaid
erDiagram
    episodes {
        text id PK
        text created_at
        text request
        text source
        text capability "last step's capability (projection)"
        text arm_id "last step's arm_id (projection)"
        text status
        int outcome_ok "last step's outcome.ok (projection)"
        real implicit_reward "mean of all steps' implicit_reward"
        int explicit_score
        text correction
        real final_reward
        text planner_id
        text policy_id
        text job_id "found by scanning steps for schedule_task"
        text session_id FK "groups episodes into a conversation, nullable"
        text data "full Episode JSON: state, steps[], answer, ..."
    }
    sessions {
        text id PK
        text created_at
        text updated_at
        text status "active | ended"
        int turn_count "completed episodes attached to this session"
        text summary "rolling summary of turns older than summarized_through"
        int summarized_through "turn_count as of the last summarization"
    }
    episode_steps {
        text episode_id PK, FK
        int step_index PK
        text capability
        text arm_id
        int outcome_ok
    }
    episodes_fts {
        text episode_id
        text request
        text correction
        text hosts
    }
    policy_state {
        text policy_id PK
        text state "JSON: policy.state_dict()"
        text updated_at
    }
    memories {
        text id PK
        text created_at
        text updated_at
        text kind
        text text
        text capability
        int support_count
        text source_episode_ids "JSON list"
        text superseded_by FK
        int active
    }
    memories_fts {
        text memory_id
        text text
    }
    apscheduler_jobs {
        text id PK
        blob job_state
        float next_run_time
    }

    episodes ||--o{ episodes_fts : "indexed by"
    episodes ||--o{ episode_steps : "one row per Episode.steps[i]"
    memories ||--o{ memories_fts : "indexed by"
    memories |o--o| memories : "superseded_by"
    episodes ||--o{ memories : "source_episode_ids (logical, not FK)"
    sessions ||--o{ episodes : "groups (session_id)"
```

`episodes_fts` and `memories_fts` are separate SQLite FTS5 virtual tables (not real
foreign-keyed tables) kept in sync by `EpisodeStore.save()` / `MemoryStore._save()`
on every write — see `core/text.py` for the shared tokenizer/ranking logic both use.
`episode_steps` exists purely so `capability_success_rate(capability)` can average
`outcome_ok` per step rather than per episode (an episode can execute the same
capability in more than one step); the full per-step detail (candidates, params,
rationale, outcome payload) lives only in `episodes.data`'s `steps` array —
`episode_steps` is a denormalized projection, rewritten wholesale (delete + insert)
on every `EpisodeStore.save()`, same as the `episodes` row's own projected columns.

`sessions` (see [MEMORY-PLAN.md](MEMORY-PLAN.md) "Session memory (v2)") groups
episodes into an explicitly started/ended conversation — `episodes.session_id` is a
plain nullable column (`NULL` for the common session-less case), not a hard foreign
key constraint, consistent with how `job_id` already works. `EpisodeStore.
list_session_episodes(session_id)` is the only session-specific query; everything
else about a session (its rolling `summary`, `turn_count`) lives on the `sessions`
row itself, updated by `core/agent.py:Agent._finish_episode`.

## Runtime composition (`api/app.py:create_app`)

```mermaid
graph TB
    Settings["Settings<br/>(core/config.py)"] --> create_app["create_app()"]
    create_app --> httpClient["httpx.AsyncClient()"]
    create_app --> store["EpisodeStore(db_path)"]
    store --> memory["MemoryStore(store.connection)"]
    store --> sessions2["SessionStore(store.connection)"]
    create_app --> registry["CapabilityRegistry"]
    httpClient --> HttpCap2["HttpCallCapability"] --> registry
    create_app --> agentScheduler["AgentScheduler(db_path, scheduled_runner)"]
    agentScheduler --> SchedCap2["ScheduleTaskCapability"] --> registry
    create_app --> SearchCap2["WebSearchCapability(DdgsSearchPort())"] --> registry
    create_app --> codeRunner["docker_available()?<br/>DockerCodeRunner : SubprocessCodeRunner"]
    codeRunner --> RunCodeCap2["RunCodeCapability"] --> registry
    create_app --> ReportCap2["GenerateReportCapability(reports_dir)"] --> registry
    create_app --> planner2["_build_planner(settings)<br/>Mock | Claude"]
    create_app --> policy2["_build_policy(settings)<br/>LinUCB | Epsilon | Greedy"]
    store -->|"load_policy_state(policy.id)"| policy2
    create_app --> distiller2["_build_distiller(settings)<br/>Mock | Claude"]
    create_app --> summarizer2["_build_summarizer(settings)<br/>Mock | Claude"]
    memory --> consolidator["Consolidator(memory, distiller)"]
    distiller2 --> consolidator
    planner2 --> agent2["Agent(...)<br/>registers AnswerCapability if absent"]
    policy2 --> agent2
    registry --> agent2
    store --> agent2
    memory --> agent2
    consolidator --> agent2
    sessions2 --> agent2
    summarizer2 --> agent2
    agent2 -.->|"forward ref: agent_box['agent']"| agentScheduler
    agent2 --> FastAPI["FastAPI app<br/>routes + lifespan"]
    FastAPI --> bgTasks["app.state.background_tasks<br/>(SSE worker tasks — api/routes.py)"]
    FastAPI --> reportsMount["Mount /reports → reports_dir<br/>(registered before the catch-all / mount)"]
```

The scheduler/agent construction has a genuine circular dependency (the scheduler
needs to call the agent when a job fires; the agent needs the scheduler to hand to
`ScheduleTaskCapability`) resolved with a one-entry mutable dict (`agent_box`) filled
in right after `Agent` is constructed — see the comment in `api/app.py`.

## Runtime modes (`core/config.py: Mode`)

```mermaid
stateDiagram-v2
    [*] --> sim: exploration everywhere,<br/>no confirmation gate
    [*] --> dev: explore read-tier only,<br/>confirm write-tier if planner asks
    [*] --> prod_strict: never explore,<br/>always confirm write-tier

    sim: sim (simulator only)
    dev: dev (default)
    prod_strict: prod_strict
```

This is enforced in exactly two places in `core/agent.py`: `_explore_allowed()`
(feeds `Policy.select`'s `explore_mask`) and `_needs_confirmation()` (the safety
gate). Nothing else in the codebase branches on `Mode`.
