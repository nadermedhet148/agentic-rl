# agentic-rl

An agent that answers requests by executing a small set of predefined
**capabilities** — `web_search`, `http_call`, `schedule_task`, `run_code`,
`generate_report` — one step at a time, and **learns which action to take
from user feedback** via a contextual bandit (not by fine-tuning the LLM).

```
"what is langfuse?"
  → plan: web_search("langfuse")            (step 1, auto-executes — read-tier)
  → observe: 5 results, mostly generic
  → plan: web_search("langfuse llm observability")   (step 2, refines its own query)
  → observe: good results
  → plan: answer("Langfuse is an open-source AI engineering platform...")  (step 3)
  → done
```

Each of those steps is proposed by an LLM planner, but **which candidate gets
executed is picked by an RL policy** (LinUCB by default) that learns over time
from 👍/👎/✏️ feedback which kind of action actually satisfies the user for a
given request shape — see [docs/PLAN.md](docs/PLAN.md) for the full rationale.

## What this is

A single Python process (FastAPI + SQLite + APScheduler — no separate
database, no worker process, no build step for the UI) that:

- Executes a small set of **capabilities** — web search, arbitrary HTTP
  calls, scheduled tasks, sandboxed Python execution, PDF report generation —
  proposed by an LLM and picked by a learned policy (see "Why a bandit, not
  just a bigger prompt?" below).
- Is **model-agnostic**: `claude` / `openai` / `google` / `mock` behind one
  `AGENTIC_RL_PLANNER` setting (`llm/providers.py`) — swap providers without
  touching any other code.
- Ships **three interchangeable RL policies** — `linucb` (default, learns
  from context), `epsilon` (context-free baseline), `greedy` (a no-learning
  control group, always takes the planner's top pick) — see
  [docs/REINFORCEMENT-LEARNING.md](docs/REINFORCEMENT-LEARNING.md).
- Remembers across four timescales: **procedural** (the bandit's learned
  weights), **semantic** (standing rules distilled from your ✏️
  corrections), **episodic** (keyword-searchable past corrections), and
  **conversation** (an explicitly started/ended session's own turns,
  summarized every 5) — see [docs/MEMORY-PLAN.md](docs/MEMORY-PLAN.md).
- Streams every step live to the web UI over Server-Sent Events, and can
  optionally trace every plan/execute/feedback call to Langfuse — see
  [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).
- Ships a **simulator** (`sim/run.py`) that proves the bandit actually beats
  "no learning" offline, in seconds, with no LLM calls and no network.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"     # add ",observability" for Langfuse tracing
copy .env.example .env                                # fill in an API key, or leave AGENTIC_RL_PLANNER=mock
run.bat                                               # starts on http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` for the web UI, or drive it directly:

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"what is langfuse?"}'
```

No API key? Set `AGENTIC_RL_PLANNER=mock` in `.env` — a small heuristic planner
(`llm/mock.py`) exercises the same code paths for free, offline.

## How a request becomes an answer

An **episode** is one request, answered by a *loop* of steps — not a single
tool call:

1. **Plan** — the planner (an LLM, or the mock heuristic) is shown the
   registered capabilities, the request, and a summary of every step already
   taken in this episode, and proposes 1–3 candidate next actions.
2. **Select** — an RL policy picks one candidate (occasionally exploring an
   alternative, in `dev`/`sim` mode, to keep learning).
3. **Confirm gate** — a side-effecting choice (a `POST`, a recurring
   `schedule_task`, running `run_code`, writing a `generate_report`) pauses
   the episode for the user to confirm before it runs.
4. **Execute → observe** — the chosen capability runs; its outcome (success,
   error, or returned data) feeds back into step 1 for the *next* step.
5. Repeat until a step executes the special **`answer`** capability — the
   agent's own signal that it has enough to reply — or a step cap is hit.

The whole loop streams live to the web UI over Server-Sent Events
(`POST /chat/stream`), so you watch it plan → act → observe → plan again in
real time, ending in a highlighted final answer.

Feedback (👍 / 👎 / ✏️ *correction*) is given once per finished episode and
updates the RL policy for **every** step that ran in it; a ✏️ correction also
gets distilled into a standing rule (`core/memory.py`) that's shown to the
planner on every future request, so a fix applies immediately — before the
bandit has even had time to converge.

By default each message is its own independent episode with no memory of
earlier ones. Click **Start session** in the UI (or `POST /sessions`) to
change that: while a session is active, the planner sees the conversation so
far — so "now make that a PDF" can refer back to what you just asked about —
and it's kept from growing unbounded by summarizing every 5 turns (see
[docs/MEMORY-PLAN.md](docs/MEMORY-PLAN.md) "Session memory (v2)"). **End
session** (or `POST /sessions/{id}/end`) goes back to session-less chat.

## Why a bandit, not just a bigger prompt?

The obvious alternative to a learned policy is: when something goes wrong,
edit the system prompt (`llm/prompts.py:SYSTEM_PROMPT`) or add a few-shot
example, and re-deploy. This project already does a version of that — every
✏️ correction gets distilled into a **standing rule**
(`core/memory.py:Consolidator`) and injected into the next prompt
(`render_rules`) — but that alone isn't enough, which is why there's a bandit
on top of it, not instead of it:

- **Rules only capture what a user can articulate.** *"Always send an
  `Accept` header to api.example.com"* is a rule someone can write down. *"Of
  two otherwise-similar candidates, the one with this param shape tends to
  work better for this kind of request"* isn't — nobody writes that
  correction. The bandit turns plain implicit signal (did the call succeed?
  did the user 👎 it?) into a numeric estimate per `(capability,
  param-shape)` arm automatically — no one has to author anything.
- **A prompt can't do calibrated exploration.** LinUCB's score is
  `θᵀx + α·√(xᵀA⁻¹x)` — a learned mean *plus* a confidence bonus that
  shrinks as an arm gets tried more (see
  [docs/REINFORCEMENT-LEARNING.md](docs/REINFORCEMENT-LEARNING.md)). That's
  a principled, bounded way to occasionally try an under-tested alternative
  without gambling on a poorly-understood one. A static prompt has no
  equivalent of "I've only seen this arm 3 times, get more signal before
  trusting it" — an LLM re-reading the same instructions either always
  plays it safe or, if told to "sometimes try something different," does so
  with no actual guarantee about how much exploration is safe.
- **Updates are O(1) and scoped, not global.** `Policy.update` is one rank-1
  matrix update (`A += xxᵀ`, `b += reward·x`) for the single arm that just
  got feedback — milliseconds, no redeploy, and it can't accidentally change
  behavior for anything else. A rewritten sentence in a shared system prompt
  has blast radius across every capability and every user at once; a bandit
  update only moves that one arm's own weights.
- **It scales without bloating the prompt.** If every learned preference had
  to become prompt text, the prompt would grow without bound, cost more per
  call, and eventually contradict itself. Numeric weights (`policy_state` —
  one row per policy, an `A`/`b` matrix pair per arm) live entirely outside
  the prompt; only the small number of already-deduplicated rules a
  distiller has merged ever reach the prompt text at all.

None of this replaces prompting — the planner still *proposes* every
candidate through its own (corrections-aware) prompt. The bandit's job is
narrower and complementary: given what the planner suggests, decide *which*
suggestion to actually trust, using a kind of signal a prompt alone can't
represent. See [docs/PLAN.md](docs/PLAN.md) for the original design decision
and [docs/REINFORCEMENT-LEARNING.md](docs/REINFORCEMENT-LEARNING.md) for
exactly how that decision gets made.

## The web UI

`http://127.0.0.1:8000` (`api/static/index.html` — one static page, vanilla
JS, no build step) has five panels:

- **Chat** (left) — the **Start/End session** control described above, the
  running conversation, and an input box. Each reply is a live-updating card:
  a per-step timeline (icon for the capability, its rationale, a collapsible
  list of every candidate the planner considered with confidence bars and a
  "chosen"/"explored" badge, its params, and its outcome), streamed in as
  each step actually runs — you watch it plan → act → observe → plan again
  in real time — ending in a highlighted answer bubble. A step that needs
  confirmation (a write-tier action) pauses the card with a **Confirm**
  button; a finished episode gets 👍 **Good** / 👎 **Bad** / ✏️ **Correct**
  (the last opens an inline box for what it should have done instead).
- **Reward** — the rolling mean reward over the last 100 episodes and a live
  chart (Chart.js) of the underlying learning curve — this is the same
  number `sim/run.py` prints when measuring the bandit offline, just live
  against whatever you've actually been doing in the UI.
- **Rules** — the standing rules distilled from your ✏️ corrections
  (`core/memory.py`), each with a "seen ×N" badge (`support_count` —
  how many times an equivalent correction has been made) and a ✕ to retire
  one. You can also add a rule directly here without first triggering a
  correction.
- **Scheduled tasks** — every job created by a `schedule_task` step, its
  next run time, and a **Cancel** button (cancelling penalizes the episode
  that originally scheduled it, not the policy directly — see
  [docs/REINFORCEMENT-LEARNING.md](docs/REINFORCEMENT-LEARNING.md)).
- **Episodes** — a table of recent episodes (request, status, step count,
  reward) refreshed every 15s, independent of what's still visible in the
  Chat panel above (that log resets on page reload; this table is the
  persisted view — `GET /episodes`).

The mode shown next to the title in the header is `AGENTIC_RL_MODE`
(`sim`/`dev`/`prod_strict`) — it's read-only in the UI, set via `.env`.

## Multiple agents that learn from each other

Point `AGENTIC_RL_AGENTS_FILE` at a JSON list of agent profiles (see
[`agents.example.json`](agents.example.json): a researcher, an integrator and an
analyst) and the one agent becomes a **team of specialists**. Each one has its own
persona, its own slice of the capabilities and its own bandit. Without the file,
nothing changes: it's the single agent described above.

- **Routing.** A request with no agent named goes to whichever agent the router
  picks. The router is a bandit over agents that learns from the same 👍/👎.
  The chat box also lets you pick an agent explicitly.
- **Learning from each other.** Each agent scores actions with its own bandit
  evidence plus its peers', weighted by **learned trust**: how well a peer's model
  predicts the rewards *this* agent actually gets. Peers that agree are trusted
  fully. A peer whose user wants the opposite loses trust on exactly those
  actions. Peers' corrections and approved episodes also reach the planner.
- **Shared rules, carefully.** A correction becomes a rule private to the agent
  that got it. It's shared team-wide once a second agent's feedback independently
  agrees. A rule that skips a confirmation is never shared automatically; the
  **Share** button in the Rules panel is the only way.
- **Delegation.** An agent can hand a sub-task to a better-suited peer
  (`delegate`). The peer's own confirm gate still applies. If it pauses, the
  delegating episode pauses too, and one **Confirm** resumes both.

The **Team** panel shows each agent, its capabilities and reward, plus the trust
matrix between agents. See [docs/MULTI-AGENT-PLAN.md](docs/MULTI-AGENT-PLAN.md)
for the design and the simulator results that back it.

## Capabilities

See [docs/CAPABILITIES.md](docs/CAPABILITIES.md) for the full writeup
(safety tiers, confirmation behavior, how to add a new one) — short version:

| Capability | Tier | What it does |
|---|---|---|
| `web_search` | read | DuckDuckGo search (`ddgs`, no API key needed) |
| `http_call` | read (GET/HEAD) / write (POST/PUT/PATCH/DELETE) | arbitrary HTTP request |
| `schedule_task` | write | schedules a natural-language instruction to re-run later (cron or one-off) |
| `run_code` | write, always confirmed | runs a Python snippet for computation/data reshaping — sandboxed in Docker (`--network none`) when Docker is installed, else a plain subprocess fallback |
| `generate_report` | write | renders text content as a PDF, served at `/reports/<filename>` |
| `answer` | read | the terminal step — the agent's own "I'm done, here's the reply" signal |
| `delegate` | read (the peer's own gates apply) | team mode only: hand a sub-task to a peer agent and use its answer |

## Learn more

Each doc below covers one layer in depth — start with whichever question
you actually have, they cross-link to each other where they overlap.

| Doc | What's in it |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | Design rationale: why RL over decisions, not weights; reward shaping; project layout |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Structural map — components, data model, package layout |
| [docs/REINFORCEMENT-LEARNING.md](docs/REINFORCEMENT-LEARNING.md) | The RL flow in depth: arms, features, policies (LinUCB/epsilon/greedy), reward shaping, how to measure that it's actually learning |
| [docs/CAPABILITIES.md](docs/CAPABILITIES.md) | Every capability in depth: tiers, confirmation behavior, per-capability notes, how to add a new one |
| [docs/SEQUENCES.md](docs/SEQUENCES.md) | Step-by-step message flows (chat loop, confirm, feedback, scheduling), as diagrams |
| [docs/TESTING.md](docs/TESTING.md) | Worked `curl` examples for every case the system handles, run against a live instance |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | Optional Langfuse tracing — what gets traced and how to turn it on |
| [docs/MEMORY-PLAN.md](docs/MEMORY-PLAN.md) | The standing-rules ("semantic memory") design |
| [docs/MULTI-AGENT-PLAN.md](docs/MULTI-AGENT-PLAN.md) | Plan for multiple specialist agents that learn from each other — pooled bandit stats, shared rules, trust, routing, delegation |

## Running the tests

```bash
.venv\Scripts\python -m pytest                 # unit tests — no network, no API key
.venv\Scripts\python -m agentic_rl.sim.run --policy linucb --episodes 300   # learning-curve simulator
.venv\Scripts\python -m agentic_rl.sim.run --scenario transfer --episodes 60   # multi-agent: sharing on vs off
.venv\Scripts\python -m agentic_rl.sim.run --scenario conflict --episodes 150  # multi-agent: users who disagree
.venv\Scripts\python -m agentic_rl.sim.run --scenario routing --episodes 300   # multi-agent: router accuracy
```
