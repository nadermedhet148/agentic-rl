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

## Learn more

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

## Running the tests

```bash
.venv\Scripts\python -m pytest                 # unit tests — no network, no API key
.venv\Scripts\python -m agentic_rl.sim.run --policy linucb --episodes 300   # learning-curve simulator
```
