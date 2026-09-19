# Sequences

Step-by-step message flows through the system. Pair with [ARCHITECTURE.md](ARCHITECTURE.md)
for what each box actually is. All diagrams use the object names as they appear in
code (`Agent`, `EpisodeStore`, ...), not the class *types* — where a `Planner` or
`Policy` is swappable, the diagram just says "Planner"/"Policy".

## 1. Chat → read-tier action → auto-executed

The common case: a GET-like request needs no confirmation and completes in one
round trip. `POST /chat` → `Agent.run(request, source="user")`.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI /chat
    participant Agent
    participant Store as EpisodeStore
    participant Memory as MemoryStore
    participant Planner
    participant Policy
    participant Registry as CapabilityRegistry
    participant Cap as HttpCallCapability

    User->>UI: types a request
    UI->>API: POST /chat {message}
    API->>Agent: run(request, source="user")
    Agent->>Store: maybe_penalize_reissue(request, "user")
    Agent->>Store: correction_count(request)
    Agent->>Store: search_corrections(request)
    Agent->>Memory: active_rules()
    Agent->>Planner: plan(state, tool_schemas, corrections, rules)
    Planner-->>Agent: [Candidate, ...]
    Agent->>Agent: build Arm per candidate (policy/features.py)
    Agent->>Agent: explore_mask per candidate (Mode-dependent)
    Agent->>Policy: select(arms, explore_mask)
    Policy-->>Agent: (index, explored)
    Agent->>Registry: get_or_none(candidate.capability)
    Registry-->>Agent: HttpCallCapability
    Agent->>Agent: tier_for(params) == READ → needs_confirm = False
    Agent->>Cap: execute(params)
    Cap-->>Agent: Outcome(ok, status, payload)
    Agent->>Agent: implicit_reward = reward.implicit_reward(...)
    Agent->>Store: save(episode)
    Agent->>Policy: update(arm, weighted_reward)
    Agent->>Store: save_policy_state(policy.id, state_dict())
    Agent-->>API: Episode(status="executed")
    API-->>UI: 200 Episode JSON
    UI-->>User: renders result + 👍/👎/✏️
```

## 2. Chat → write-tier action → confirm

A side-effecting action (e.g. `POST`/`DELETE` `http_call`, or any `schedule_task`)
is proposed but not executed until the user confirms. The episode is persisted at
`pending_confirmation` and re-fetched by `Agent.confirm`.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI
    participant Agent
    participant Store as EpisodeStore
    participant Registry as CapabilityRegistry
    participant Cap as Capability

    User->>UI: "place an order at ..."
    UI->>API: POST /chat {message}
    API->>Agent: run(request, "user")
    Note over Agent: plan → select, same as sequence 1
    Agent->>Agent: tier == WRITE and<br/>(candidate.needs_confirmation or mode==prod_strict)
    Agent->>Store: save(episode, status="pending_confirmation")
    Agent-->>API: Episode(status="pending_confirmation", outcome=null)
    API-->>UI: 200 Episode JSON
    UI-->>User: shows "Confirm" button

    User->>UI: clicks Confirm
    UI->>API: POST /confirm/{episode_id}
    API->>Agent: confirm(episode_id)
    Agent->>Store: get(episode_id)
    Store-->>Agent: Episode(status="pending_confirmation")
    Agent->>Registry: get_or_none(candidate.capability)
    Agent->>Cap: execute(params)
    Cap-->>Agent: Outcome
    Agent->>Agent: implicit_reward, status="executed"
    Agent->>Store: save(episode)
    Agent->>Agent: recompute Arm from episode.state
    Agent->>Store: save_policy_state(...)
    Agent-->>API: Episode(status="executed")
    API-->>UI: 200 Episode JSON
    UI-->>User: renders result
```

If the episode is not `pending_confirmation` (already executed, or unknown id),
`Agent.confirm` raises `KeyError` → `404`, or `ValueError` → `409` (see `api/routes.py`).

## 3. Feedback with a correction → policy update + memory consolidation

This is where the two learning mechanisms meet: `Policy.update` (procedural
memory) and `Consolidator.consolidate` (semantic memory) both fire off one
`POST /feedback` call, in that order.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI /feedback
    participant Agent
    participant Store as EpisodeStore
    participant Policy
    participant Consolidator
    participant Memory as MemoryStore
    participant Distiller

    User->>UI: clicks 👎 / ✏️, types a correction
    UI->>API: POST /feedback {episode_id, score, correction}
    API->>Agent: record_feedback(feedback)
    Agent->>Store: apply_feedback(feedback)
    Note over Store: explicit_score, correction set<br/>final_reward = explicit_reward(...) (±1, correction forces -1)
    Store-->>Agent: updated Episode
    Agent->>Agent: rebuild Arm from episode.state + candidate
    Agent->>Policy: update(arm, weighted_reward)
    Note over Policy: explicit feedback weighted ×3<br/>vs. implicit-only updates (rl/reward.py)
    Agent->>Store: save_policy_state(...)

    alt feedback.correction is set
        Agent->>Consolidator: consolidate(correction, episode)
        Consolidator->>Memory: search(correction, limit=5)
        Memory-->>Consolidator: candidate existing rules
        Consolidator->>Memory: active_rules(capability=episode.capability)
        Memory-->>Consolidator: more candidates
        Consolidator->>Distiller: distill(correction, episode, existing)
        Distiller-->>Consolidator: DistillResult(rule_text, matches_existing_id?, supersedes_id?)
        Consolidator->>Consolidator: drop any id not in `existing` (hallucination guard)
        alt matches_existing_id (valid)
            Consolidator->>Memory: bump_support(id, episode.id)
        else supersedes_id (valid)
            Consolidator->>Memory: add(new rule)
            Consolidator->>Memory: supersede(old_id, new_id)
        else
            Consolidator->>Memory: add(new rule)
        end
        Consolidator-->>Agent: Memory
    end

    Agent-->>API: updated Episode
    API-->>UI: 200 Episode JSON
    UI-->>User: reward/rules panels refresh
```

## 4. Scheduled task: create → fire → learn

`schedule_task` is a capability like any other — it goes through the same plan
→ select → execute path as sequence 1/2. What's specific to it is what happens
*after* `execute`: a real job is registered with APScheduler, and firing that job
re-enters `Agent.run` on a background thread.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI
    participant Agent
    participant Cap as ScheduleTaskCapability
    participant Sched as AgentScheduler
    participant APS as APScheduler<br/>(background thread)
    participant EvLoop as asyncio event loop<br/>(main thread)

    User->>UI: "every day at 9am fetch ..."
    UI->>API: POST /chat
    API->>Agent: run(request, "user")
    Note over Agent: plan → select (candidate.capability == "schedule_task")
    Agent->>Cap: execute({instruction, cron/run_at, timezone})
    Cap->>Sched: add_job(instruction, cron=..., timezone=...)
    Sched->>APS: scheduler.add_job(_run_scheduled_instruction, trigger, args=[instruction])
    APS-->>Sched: job.id
    Sched-->>Cap: job_id
    Cap-->>Agent: Outcome(ok=True, payload={job_id})
    Note over Agent,API: episode saved with job_id column<br/>(EpisodeStore.save, for later cancellation)
    Agent-->>API: Episode(status="executed")
    API-->>UI: 200, job now listed in GET /tasks

    Note over APS: ... time passes, cron/run_at fires ...
    APS->>APS: _run_scheduled_instruction(instruction)<br/>(module-level fn, picklable job ref)
    APS->>Sched: _dispatch(instruction)
    Sched->>EvLoop: run_coroutine_threadsafe(agent.run(instruction, source=scheduler))
    EvLoop->>Agent: run(instruction, source="scheduler")
    Note over Agent: same plan → select → execute path,<br/>state.source="scheduler" (feeds Arm features)
    Agent->>Agent: finalize_execution(...) as usual
```

### Cancellation

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI
    participant Sched as AgentScheduler
    participant Agent
    participant Store as EpisodeStore

    User->>UI: clicks Cancel on a task
    UI->>API: DELETE /tasks/{job_id}
    API->>Sched: remove_job(job_id)
    Sched->>Sched: scheduler.remove_job(job_id)
    API->>Agent: cancel_task(job_id)
    Agent->>Store: mark_task_cancelled(job_id)
    Store->>Store: find episode by job_id column,<br/>implicit_reward += CANCELLED (-0.5)
    Store-->>Agent: penalized Episode
    API-->>UI: 200 {"status": "cancelled"}
```

Cancelling a task never touches the policy directly — the penalty lands on the
*episode that originally scheduled it*, so the next time the same kind of
`schedule_task` candidate is proposed, its `capability_success_rate` and (once
feedback lands) its arm's learned mean both reflect that it was cancelled.

## 5. App startup: loading persisted state

Both halves of memory that need to survive a restart — the bandit's learned
weights and the standing rules — come from the same SQLite file, loaded once in
`create_app` before the server starts accepting requests.

```mermaid
sequenceDiagram
    participant App as create_app()
    participant Store as EpisodeStore
    participant Memory as MemoryStore
    participant Policy
    participant Sched as AgentScheduler

    App->>Store: EpisodeStore(db_path)
    Store->>Store: executescript(schema)<br/>_ensure_fts_schema() (migrate if needed)
    App->>Memory: MemoryStore(store.connection)
    Note over Store,Memory: same sqlite3.Connection —<br/>one file, one transaction boundary
    App->>Policy: _build_policy(settings)
    App->>Store: load_policy_state(policy.id)
    Store-->>App: dict | None
    alt state found
        App->>Policy: load_state(state)
    end
    App->>Sched: AgentScheduler(db_path, scheduled_runner)
    Note over Sched: APScheduler's own SQLAlchemyJobStore<br/>reads its jobs table from the same file
    App->>App: Agent(planner, policy, registry, store,<br/>settings, memory, consolidator)
    Note over App: FastAPI lifespan startup:<br/>scheduler.set_loop(running_loop), then scheduler.start()
```

`MemoryStore`'s rules need no explicit load step — `Agent.run` calls
`memory.active_rules()` fresh on every request, so whatever was persisted is simply
read directly off disk the first time it's needed.
