from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agentic_rl.core import text as text_util
from agentic_rl.core.models import Episode, Feedback
from agentic_rl.rl import reward as reward_mod

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    request TEXT NOT NULL,
    source TEXT NOT NULL,
    capability TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome_ok INTEGER,
    implicit_reward REAL NOT NULL,
    explicit_score INTEGER,
    correction TEXT,
    final_reward REAL,
    planner_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    job_id TEXT,
    session_id TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_arm ON episodes(arm_id);
CREATE INDEX IF NOT EXISTS idx_episodes_capability ON episodes(capability);
CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_job_id ON episodes(job_id);
CREATE TABLE IF NOT EXISTS episode_steps (
    episode_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    capability TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    outcome_ok INTEGER,
    PRIMARY KEY (episode_id, step_index)
);
CREATE INDEX IF NOT EXISTS idx_episode_steps_capability ON episode_steps(capability);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    episode_id UNINDEXED, request, correction, hosts
);
CREATE TABLE IF NOT EXISTS policy_state (
    policy_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# bm25() weight per indexed column, in declaration order (UNINDEXED columns are
# skipped): request, correction, hosts. Hosts weighted highest — two requests that
# hit the same host are a much stronger recall signal than two that merely share
# a common word like "data".
_BM25_WEIGHTS = (1.0, 0.5, 3.0)


class EpisodeStore:
    """SQLite-backed episode log: the operational record the policy learns from and
    the source of both prompt corrections (via FTS5) and the fine-tuning export
    (rl/export.py)."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ensure_fts_schema()
        self._ensure_session_id_column()

    def _ensure_session_id_column(self) -> None:
        """Dev-project migration: episodes created before session support (see
        core/session.py) lack the session_id column — SQLite supports adding a
        column in place, no rebuild needed (unlike the FTS5 virtual table below).
        The index is created here too (not in _SCHEMA) since `CREATE INDEX ... ON
        episodes(session_id)` would fail outright against a pre-existing table that
        doesn't have the column yet — this runs after the column is guaranteed to
        exist, for both fresh and migrated databases."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(episodes)").fetchall()}
        if "session_id" not in columns:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN session_id TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id)")
        self._conn.commit()

    def _ensure_fts_schema(self) -> None:
        """Dev-project migration: if episodes_fts predates the `hosts` column, drop
        and rebuild it from the episodes table rather than carrying a migration
        framework for a throwaway SQLite file."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(episodes_fts)").fetchall()}
        if "hosts" in columns:
            return
        self._conn.execute("DROP TABLE IF EXISTS episodes_fts")
        self._conn.execute(
            "CREATE VIRTUAL TABLE episodes_fts USING fts5(episode_id UNINDEXED, request, correction, hosts)"
        )
        rows = self._conn.execute("SELECT id, request, correction FROM episodes").fetchall()
        for row in rows:
            hosts = " ".join(text_util.extract_hosts(row["request"]))
            self._conn.execute(
                "INSERT INTO episodes_fts (episode_id, request, correction, hosts) VALUES (?, ?, ?, ?)",
                (row["id"], row["request"], row["correction"] or "", hosts),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, shared with MemoryStore (core/memory.py) so a
        `:memory:` database stays a single database across both stores."""
        return self._conn

    def save(self, episode: Episode) -> None:
        last = episode.steps[-1] if episode.steps else None
        job_id = None
        for step in episode.steps:
            if (
                step.action.candidate.capability == "schedule_task"
                and step.outcome is not None
                and step.outcome.ok
                and isinstance(step.outcome.payload, dict)
            ):
                job_id = step.outcome.payload.get("job_id")
                break

        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, created_at, request, source, capability, arm_id, status, outcome_ok,
                implicit_reward, explicit_score, correction, final_reward, planner_id,
                policy_id, job_id, session_id, data)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                episode.id,
                episode.created_at.isoformat(),
                episode.state.request,
                episode.state.source,
                last.action.candidate.capability if last else "",
                last.action.arm_id if last else "",
                episode.status,
                None if last is None or last.outcome is None else int(last.outcome.ok),
                episode.implicit_reward,
                episode.explicit_score,
                episode.correction,
                episode.final_reward,
                episode.planner_id,
                episode.policy_id,
                job_id,
                episode.session_id,
                episode.model_dump_json(),
            ),
        )
        self._conn.execute("DELETE FROM episode_steps WHERE episode_id = ?", (episode.id,))
        self._conn.executemany(
            "INSERT INTO episode_steps (episode_id, step_index, capability, arm_id, outcome_ok) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    episode.id,
                    step.index,
                    step.action.candidate.capability,
                    step.action.arm_id,
                    None if step.outcome is None else int(step.outcome.ok),
                )
                for step in episode.steps
            ],
        )
        hosts = " ".join(text_util.extract_hosts(episode.state.request))
        self._conn.execute("DELETE FROM episodes_fts WHERE episode_id = ?", (episode.id,))
        self._conn.execute(
            "INSERT INTO episodes_fts (episode_id, request, correction, hosts) VALUES (?, ?, ?, ?)",
            (episode.id, episode.state.request, episode.correction or "", hosts),
        )
        self._conn.commit()

    def get(self, episode_id: str) -> Episode | None:
        row = self._conn.execute("SELECT data FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return Episode.model_validate_json(row["data"]) if row else None

    def apply_feedback(self, feedback: Feedback) -> Episode:
        episode = self.get(feedback.episode_id)
        if episode is None:
            raise KeyError(f"unknown episode: {feedback.episode_id}")
        episode.explicit_score = feedback.score
        episode.correction = feedback.correction
        episode.final_reward = reward_mod.final_reward(
            episode.explicit_score, episode.correction, episode.implicit_reward
        )
        self.save(episode)
        return episode

    def search_corrections(self, request: str, limit: int = 5) -> list[str]:
        """Corrections from past episodes whose request text overlaps this one —
        rendered into the planner prompt so a fix applies on the very next similar
        request (see llm/prompts.py:render_corrections).

        Ranking is bm25 over (request, correction, hosts) with hosts weighted
        highest (_BM25_WEIGHTS) so two requests to the same host outrank two that
        merely share a filler word. FTS5 itself can't express "at least N shared
        terms", so a min-token-overlap filter runs in Python after the ranked
        fetch — a single shared stopword-free token isn't enough to count as
        "similar" unless the query itself only has one token to begin with.
        """
        query_tokens = text_util.tokenize(request)
        query = text_util.fts_query(query_tokens)
        if not query:
            return []
        required_overlap = text_util.min_overlap(query_tokens)
        weights = ", ".join(str(w) for w in _BM25_WEIGHTS)
        rows = self._conn.execute(
            f"""SELECT e.correction AS correction, e.request AS request,
                       bm25(episodes_fts, {weights}) AS score
                FROM episodes_fts f JOIN episodes e ON e.id = f.episode_id
                WHERE episodes_fts MATCH ? AND e.correction IS NOT NULL AND e.correction != ''
                ORDER BY score LIMIT ?""",
            (query, max(limit * 4, 20)),
        ).fetchall()

        results = []
        for row in rows:
            if len(query_tokens & text_util.tokenize(row["request"])) >= required_overlap:
                results.append(f"for a request like '{row['request']}': {row['correction']}")
            if len(results) >= limit:
                break
        return results

    def correction_count(self, request: str, limit: int = 1000) -> int:
        return len(self.search_corrections(request, limit=limit))

    def capability_success_rate(self, capability: str, default: float = 0.5) -> float:
        row = self._conn.execute(
            "SELECT AVG(outcome_ok) AS rate FROM episode_steps WHERE capability = ? AND outcome_ok IS NOT NULL",
            (capability,),
        ).fetchone()
        return default if row is None or row["rate"] is None else float(row["rate"])

    def rolling_reward(self, n: int = 100) -> list[float]:
        """Most-recent-first N episodes' reward (final if feedback was given, else
        implicit), returned in chronological order for plotting a learning curve."""
        rows = self._conn.execute(
            "SELECT COALESCE(final_reward, implicit_reward) AS r FROM episodes "
            "ORDER BY created_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [row["r"] for row in reversed(rows)]

    def maybe_penalize_reissue(self, request: str, source: str, window_seconds: int = 600) -> None:
        """If the same request came from the same source within the last `window_seconds`,
        apply the REISSUED implicit penalty to that *prior* episode — re-asking shortly
        after usually means the earlier action didn't satisfy the user. Best-effort exact
        text match; a v1 simplification (see docs/PLAN.md)."""
        cutoff = (datetime.now(UTC) - timedelta(seconds=window_seconds)).isoformat()
        row = self._conn.execute(
            """SELECT id FROM episodes WHERE request = ? AND source = ? AND created_at >= ?
               ORDER BY created_at DESC LIMIT 1""",
            (request, source, cutoff),
        ).fetchone()
        if row is None:
            return
        episode = self.get(row["id"])
        if episode is None:
            return
        episode.implicit_reward += reward_mod.REISSUED
        if episode.explicit_score is None:
            episode.final_reward = episode.implicit_reward
        self.save(episode)

    def mark_task_cancelled(self, job_id: str) -> Episode | None:
        """Apply the CANCELLED implicit penalty to the episode that scheduled `job_id`."""
        row = self._conn.execute("SELECT id FROM episodes WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        episode = self.get(row["id"])
        if episode is None:
            return None
        episode.implicit_reward += reward_mod.CANCELLED
        if episode.explicit_score is None:
            episode.final_reward = episode.implicit_reward
        self.save(episode)
        return episode

    def save_policy_state(self, policy_id: str, state: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO policy_state (policy_id, state, updated_at) VALUES (?, ?, ?)",
            (policy_id, json.dumps(state), datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def load_policy_state(self, policy_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT state FROM policy_state WHERE policy_id = ?", (policy_id,)
        ).fetchone()
        return json.loads(row["state"]) if row else None

    def list_episodes(self, limit: int = 50) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT data FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Episode.model_validate_json(row["data"]) for row in rows]

    def list_session_episodes(self, session_id: str, limit: int = 50) -> list[Episode]:
        """Episodes attached to `session_id`, oldest first — used to pull the turns
        not yet folded into a session's rolling summary (core/agent.py:_finish_episode)."""
        rows = self._conn.execute(
            "SELECT data FROM episodes WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [Episode.model_validate_json(row["data"]) for row in rows]
