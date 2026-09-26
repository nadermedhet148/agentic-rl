# Multi-agent — plan

## Context

Today there is exactly one `Agent` (core/agent.py): one planner, one policy, one
`MemoryStore`, one `EpisodeStore`, all wired once in `api/app.py:create_app`. It
learns on three timescales ([MEMORY-PLAN.md](MEMORY-PLAN.md)):

| Tier | Where it lives today | What it learns |
|---|---|---|
| **Procedural** | `LinUCBPolicy` `A`/`b` per arm → `policy_state` row keyed by `policy_id` | which *kind* of action satisfies the user for a request shape |
| **Semantic** | `memories` table (rules), injected into every plan | the user's standing instructions ("always send Accept: application/json …") |
| **Episodic** | `episodes` + `episodes_fts`, `search_corrections()` | recent similar mistakes and their fixes |

Goal: run **several agents** that each learn from their own feedback, and that
**learn from each other**, so what agent A was taught doesn't have to be relearned
from scratch by agent B, while bad or irrelevant lessons don't spread.

The system already has the right shape for this. None of the learning happens in
LLM weights. It happens in three small, inspectable, mergeable stores: bandit
statistics, text rules, and an episode log. "Agents learning from each other"
therefore means **sharing those three stores in a controlled way**. It doesn't mean
agents chatting to each other and hoping something sticks.

### Decisions made with the user

- **One user for now.** All agents serve the same person, so a lesson confirmed by
  two agents is safe to share team-wide. Per-user rule scoping is out of scope until
  there's more than one user.
- **Delegation ships in v1** (phase 5 below), not deferred.
- **Distributed later, designed now.** Agents run in one process for now, but
  phase 2 designs the `KnowledgeHub` around a sync interface that serializes
  cleanly. That way, moving agents into separate processes later is a transport
  change, not a redesign.
- Agents are **specialists**, not clones. Each one has a role, a persona prompt and
  a *subset* of the capability registry, e.g. `researcher` (web_search, answer),
  `integrator` (http_call, schedule_task, answer) and `analyst` (run_code,
  generate_report, answer). Replicas of one generalist fall out as a special case:
  same profile, different ids.
- Still **one process, one SQLite file**. Cross-process or cross-machine sharing is
  a later phase. The sharing interfaces below are designed so they could become an
  HTTP sync later without changing the learning math.
- The existing safety rules are non-negotiable per agent. Exploration stays masked
  by mode and tier, and the write-tier confirm gate stays. **Nothing learned from a
  peer may loosen a safety gate.**

## Design

```
                           ┌──────────────── Team ────────────────┐
 POST /chat ──► Router ────┤  (bandit over agents; learns who     │
   (agent_id?)  (policy)   │   should handle which request shape) │
                           └──┬───────────────┬───────────────┬───┘
                              ▼               ▼               ▼
                        Agent "researcher" Agent "integrator" Agent "analyst"
                        planner+persona    planner+persona    planner+persona
                        caps ⊂ registry    caps ⊂ registry    caps ⊂ registry
                        local policy       local policy       local policy
                              │  ▲            │  ▲            │  ▲
                   publish ───┘  └── pull ────┘  └────────────┘  │
                              ▼                                  │
                   ┌──────────────── KnowledgeHub ───────────────┴──┐
                   │ procedural: per-agent bandit stats, trust-      │
                   │             weighted pooling (LinUCB is additive)│
                   │ semantic:   rules with owner + scope, promotion │
                   │             private → team after cross-agent    │
                   │             confirmation                        │
                   │ episodic:   peer corrections + peer *successes* │
                   │             (positive demonstrations)           │
                   │ trust:      w[i][j] = how well j's knowledge    │
                   │             predicts i's rewards                │
                   └─────────────────────────────────────────────────┘
                   + `delegate` capability: an agent hands a sub-task to a peer
```

There are four channels of "learning from each other", from cheapest and most
principled to richest:

### 1. Procedural sharing: pooled bandit statistics

Disjoint LinUCB's sufficient statistics are **additive**. For an arm, `A = I + Σ xxᵀ`
and `b = Σ r·x`. If agent *j* saw the same arm, its evidence is `(A_j − I, b_j)`,
and adding it to agent *i*'s matrices is exactly what *i* would have learned from
seeing *j*'s data itself. That makes this the cleanest transfer channel in the
system.

- Each agent keeps **only its own local statistics** (`A_i`, `b_i`). They're
  persisted per agent, so `policy_state` becomes keyed by `(agent_id, policy_id)`.
- At `select()` time the agent scores with an **effective** model:

  ```
  A_eff = I + (A_i − I) + Σ_{j≠i} w_ij · (A_j − I)
  b_eff =      b_i      + Σ_{j≠i} w_ij ·  b_j
  ```

  This is recomputed from everyone's local stats, never accumulated into them. That
  makes pooling **idempotent**: syncing twice can't double-count evidence, which is
  the classic bug in naive "merge the models" schemes. At 32×32 floats per arm and a
  handful of agents, recomputing is trivially cheap. It's cached per arm and
  invalidated when any contributor updates.
- `w_ij ∈ [0, 1]` is the **trust** agent *i* places in *j* (channel 4 below). With
  `w = 0` for all peers, behaviour is exactly today's single agent. That gives a
  built-in control group.
- Arms are keyed by `capability:param_template:needs_confirmation`
  (policy/features.py), so two agents only pool on arms they *both can take*. A
  researcher's `web_search` evidence never touches an integrator that doesn't have
  `web_search`.
- `EpsilonGreedyPolicy` pools the same way: count-weighted means with `w_ij`-scaled
  counts. `GreedyPolicy` has nothing to share. It stays the no-learning control.
- New abstract method on `Policy`: `evidence() -> dict` (local-only stats) and
  `set_peer_evidence(peers: list[tuple[float, dict]])`, next to the existing
  `state_dict`/`load_state` (policy/base.py).

**Negative transfer guard:** agents may serve different users with different
preferences, for example two timezones. So `w_ij` is *learned*, not fixed, and
features can include an agent/role one-hot slot (spare hashed dims exist in
`FEATURE_DIM`). That way a shared arm can still diverge per agent where rewards
genuinely differ.

### 2. Semantic sharing: rules with provenance and promotion

`Memory` gains `owner_agent_id` and `scope: "private" | "team"`.

- A correction on agent *i*'s episode is consolidated exactly as today
  (`Consolidator`), into a rule **owned by *i*, scope `private`**.
- The planner prompt gets *i*'s private rules plus all active `team` rules
  (`llm/prompts.py:render_rules`, with team rules labelled as such).
- **Promotion to `team`** happens when a second agent's feedback independently
  supports the same rule, via the Distiller's existing `matches_existing_id` path,
  now searched across all agents' rules. It can also be promoted by a user through
  the API. `support_count` is split into per-agent support so "5 corrections from
  one agent" doesn't masquerade as team-wide consensus.
- **Scope check:** a rule with `capability` set is only shown to agents that have
  that capability.
- **Safety filter:** a peer rule that would *remove* a confirmation or widen a
  write-tier action ("don't ask before POSTing") is never auto-promoted. It stays
  private to its owner until a human promotes it. Rules that *add* caution promote
  freely. This is asymmetric on purpose.
- Contradictions across agents use the existing `supersede` mechanism, but only
  within the same scope. A private rule can override a team rule *for its owner*.
  It can't retire the team rule for everyone.

### 3. Episodic sharing: peer corrections and peer demonstrations

- `search_corrections()` gains an `agent_id` filter. It searches the agent's own
  episodes first, then peers' episodes, weighted by trust, and labels them in the
  prompt as coming from a peer.
- **New: positive demonstrations.** Today only *mistakes* flow back into the prompt.
  In a team, the most useful thing A can give B is "here is a request like yours
  that I handled and the user 👍'd": the trajectory's capabilities, param shapes
  and answer. So we add `search_demonstrations(request, agent_id)`, which returns
  peers' `final_reward > 0` episodes whose capabilities the current agent actually
  has. They're rendered as few-shot examples (`render_demonstrations` in
  llm/prompts.py). It's cheap, needs no new dependency, and the episode log already
  stores everything (`Episode.steps`, `data` JSON column).
- This log is also the future fine-tuning set (rl/export.py). Tagging every episode
  with `agent_id` makes per-role DPO datasets fall out for free.

### 4. Trust: learning *whom* to learn from

`w_ij` is estimated online, per ordered pair **and per arm**, with a pair-level
estimate as the fallback for arms with too few observations. After agent *i*
receives a reward `r` on arm `a`, compare that with the prediction *j*'s model
alone would have made for `a` on the same features (`θ_jᵀx`), and with *i*'s own
out-of-sample prediction. Peers whose predictions track *i*'s actual rewards earn
trust. Peers that mislead lose it:

```
err_peer ← EMA_β of (r − θ_jᵀx)²     # j's prediction error on i's rewards
err_self ← EMA_β of (r − θ_iᵀx)²     # i's own error, before it updates on r
w_ij     = clip(((err_self + ε) / (err_peer + ε))², 0, 1)
```

A peer that predicts *i*'s rewards as well as *i* itself gets 1. The fall-off is
squared because a peer usually has far more evidence on an arm than *i* does, and a
linear fall-off left enough of a conflicting peer's weight to outvote *i*'s own
early evidence. This was tuned on the conflict scenario (see Results).

Trust starts at a configurable prior (`trust_prior`, default 0.5) until there are
`trust_min_obs` (default 2) observations. It's stored in an `agent_trust` table so
it survives restarts and is visible in the UI.
The same weights gate channels 2 and 3 (peer rules and peer corrections are ranked
by owner trust), so one number per pair drives all sharing.

### Routing and delegation

- **Router** (`core/router.py`): when `/chat` arrives without an `agent_id`, a bandit
  picks the agent. Arms are agent ids, and context is a hashed bag of the request's
  tokens and hosts. It learns from the *same* episode feedback, so if the analyst
  keeps getting 👎 on "fetch …" requests, those drift to the integrator. It reuses
  `LinUCBPolicy` unchanged, plus a capability-cue prior ("this request mentions a
  URL, and this agent has `http_call`") that fades as real feedback accumulates, so
  routing is sensible from the first request. An explicit `agent_id` bypasses it.
- **Delegation** (`capabilities/delegate.py`): a new capability,
  `delegate(agent_id, request)`. It runs a peer's `Agent.run` as a child episode
  (`parent_episode_id`) and returns its answer as the outcome payload. Delegating
  itself is read-tier, because everything the child does goes through the *child's*
  own confirm gate. If the child pauses there, the parent pauses too, so delegation
  can't launder a write past confirmation. Confirming the parent confirms the
  child, and the child's completion resumes the parent. Depth is capped
  (`max_delegation_depth`, default 1) to avoid ping-pong.
- **Credit assignment:** feedback on the parent updates the parent's arms (including
  the `delegate:<peer>` arm, so agents *learn whom to ask*). It also propagates to
  the child episode at a discount (`delegation_credit`, default 0.5), so the peer
  learns too.

## Changes

### 0. Plumbing, with no behavior change ✅ done

- `core/models.py`: `AgentProfile(id, name, role, persona, capabilities: list[str],
  policy: str, share: bool)`. Add `agent_id` to `Episode` (default `"default"`),
  `owner_agent_id` + `scope` to `Memory`, and `parent_episode_id` to `Episode`.
- `core/store.py`: `episodes.agent_id` column + index (same in-place `ALTER TABLE`
  migration pattern as `_ensure_session_id_column`). Re-key `policy_state` by
  `(agent_id, policy_id)`, migrating the existing row to `agent_id="default"`. New
  `agents` and `agent_trust` tables.
- `core/memory.py`: `owner_agent_id`, `scope` columns (migrate existing rules to
  `owner="default", scope="team"`, which preserves today's behaviour).
- `capabilities/registry.py`: `view(names) -> CapabilityRegistry`, a filtered view
  so each agent's `tool_schemas()` only lists its own capabilities.
- `Agent.__init__` takes an `AgentProfile`, and `_persist_policy` writes under its id.
- The `agents` and `agent_trust` tables are deferred to phases 1 and 4, where they're
  first read.
- **Acceptance:** the whole existing test suite passes unchanged with one implicit
  `default` agent.

### 1. Multiple agents + Team + Router ✅ done

- `core/team.py`: `Team(agents: dict[str, Agent], router: Policy, hub: KnowledgeHub)`
  with `run(request, agent_id=None, …)`, `confirm`, `record_feedback`. It delegates
  to the owning agent, found via `episode.agent_id`.
- `llm/prompts.py`: the persona is prepended to the system prompt.
- `core/config.py`: `agents_file: Path | None`. This is a JSON list of profiles
  (see `agents.example.json`). When unset, a single `default` agent keeps today's setup.
- `api/app.py`: build the `Team` instead of a single `Agent`. `app.state.agent` stays
  as the default agent for back-compat.
- `api/routes.py`: optional `agent_id` on `/chat`, `/chat/stream`; `GET /agents`,
  `GET /agents/{id}/metrics`; `agent_id` filter on `/episodes`, `/memories`.

### 2. Procedural sharing (the core experiment) ✅ done

- `core/hub.py`: `KnowledgeHub` holds references to every agent's policy and the
  trust table. `refresh(agent_id)` computes peer evidence and calls
  `set_peer_evidence`. It runs lazily just before an agent selects, and is a
  no-op unless some agent's evidence or trust changed since its last refresh.
- `policy/linucb.py`, `policy/epsilon.py`: `evidence()` /
  `set_peer_evidence()`, plus an effective-model cache.
- Trust starts at a fixed prior. Learned trust comes in phase 4.

### 3. Semantic + episodic sharing ✅ done

- `Consolidator`: cross-agent match search, per-agent support, the promotion rule,
  and the safety filter (a small, explicitly tested classifier for "loosens
  confirmation": Distiller output field `loosens_safety: bool`, and a mock
  heuristic for the mock distiller).
- `EpisodeStore.search_corrections(…, agent_id, peer_weights)` and a new
  `search_demonstrations`.
- `Planner.plan(…, demonstrations: list[str] | None = None)` → `render_demonstrations`.
- `POST /memories/{id}/promote`, `POST /memories/{id}/demote`.

### 4. Learned trust ✅ done

- `core/hub.py`: trust EMA update inside `record_feedback` / `_finalize_step`, then
  persisted to `agent_trust`.
- `GET /agents/trust` returns the matrix for the UI.

### 5. Delegation ✅ done

- `capabilities/delegate.py`, `Episode.parent_episode_id`, discounted credit
  propagation in `Team.record_feedback`, the confirm-gate pass-through, and the
  depth cap.

### 6. UI ✅ done

- An agent selector (or "auto") in the chat box. The episode list shows the agent
  and any delegation chain. The memories panel shows scope/owner with a promote
  button. There's a small trust heatmap and a per-agent reward curve on the metrics
  panel.

## Verification: proving they actually learn from each other

The simulator (sim/run.py) is extended rather than replaced. It already gives a
clean bad/good learning signal in seconds without an LLM.

1. **Transfer test.** Two agents with the same capabilities. Agent A trains alone on
   the three `REQUESTS` for N episodes. Then agent B (fresh) starts on the same
   requests. Metric: B's mean reward over its first 50 episodes with sharing
   `w = prior` vs `w = 0`. **Expectation:** with sharing, B starts near A's final
   reward instead of at the greedy baseline.
2. **No negative transfer.** Two agents whose scripted users *disagree*: one user
   wants to confirm orders, the other finds that annoying
   (`ScriptedUser(order_confirmation=False)`). Timezone was the first idea, but a
   timezone *value* isn't part of an arm's identity (only its param shape is), so
   the bandit can't tell Berlin from Tokyo; that preference is the rules' job.
   **Expectation:** learned trust drops toward 0 on the order arms, stays high on
   the arms they agree on, and both agents converge to *their own* user's
   preference. Their final reward isn't worse than the `w = 0` control's.
3. **Rule promotion.** Both agents get the same correction once. **Expectation:**
   one `team` rule with per-agent support {A:1, B:1}, not two private duplicates. A
   rule that removes a confirmation stays private.
4. **Routing.** Three specialists and mixed requests. **Expectation:** the router's
   pick accuracy (vs. the scripted "right specialist") rises above chance within a
   few hundred episodes.
5. **Delegation safety.** A read-only agent delegates to one whose child step is
   write-tier. **Expectation:** the parent episode ends in `pending_confirmation`,
   never auto-executes.

`python -m agentic_rl.sim.run --scenario {transfer,conflict,routing} --share
{on,off,both}` prints the curves (sim/team.py). tests/test_team.py asserts every
expectation above, the same approach as tests/test_learning.py.

### Results

From `sim/team.py` with the mock planner (deterministic):

| Scenario | Sharing on | Sharing off |
|---|---|---|
| Transfer: fresh agent `b`, mean reward over its first 15 episodes | **1.00** | 0.73 |
| Conflict: mean reward over first 30 rounds, `a` / `b` | 0.80 / 0.87 | 0.87 / 0.80 |
| Conflict: mean reward over last 30 rounds, `a` / `b` | **1.00 / 1.00** | 1.00 / 1.00 |
| Conflict: learned trust `b → a` on the order-confirm arm / on the data arm | < 0.2 / > 0.8 | n/a |
| Routing: router pick accuracy, first 100 → last 100 episodes (cue prior off) | 0.97 → 1.00 | n/a |

With a linear trust curve, the conflict scenario's early rounds cost `b` about 0.4
(0.40 vs 0.80). The squared curve brings the early rounds to parity, with the two
agents' combined early reward matching the control's, while keeping the full
transfer benefit.

## Rollout order and scope

Phases 0 → 2 are the minimum that delivers "agents learn from each other" in a
measurable way (test 1). Phase 3 adds the human-readable channel (rules and
demonstrations). Phase 4 makes sharing safe under heterogeneous users (test 2).
Phase 5 adds collaboration on a single request. Each phase is independently
shippable and keeps `share=false` / single-`default`-agent behaviour identical to
today.

Out of scope for now: cross-process/federated sync (the `evidence()` /
`set_peer_evidence()` interface is the seam for it later), LLM-to-LLM debate or
chat between agents, and per-agent fine-tuning.

## Resolved questions

1. **Specialists or replicas?** Specialists, as designed above.
2. **One user or many?** One user for now.
3. **Delegation in v1?** Yes.
4. **Distributed later?** Yes, eventually. The hub's sync interface is designed in
   phase 2 so it can later run over a network.
