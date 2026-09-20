from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from agentic_rl.core.models import Session

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    turn_count INTEGER NOT NULL,
    summary TEXT NOT NULL,
    summarized_through INTEGER NOT NULL
);
"""


class SessionStore:
    """SQLite-backed conversation sessions — explicitly started/ended by the user
    (see api/routes.py POST /sessions, POST /sessions/{id}/end), not an always-on
    notion. core/agent.py injects an active session's summary + recent turns into
    every plan, and folds completed turns into the summary every
    settings.session_summarize_every turns.

    Shares its connection with EpisodeStore (via EpisodeStore.connection), same
    pattern as MemoryStore (core/memory.py), so a `:memory:` database stays one
    database across all three stores.
    """

    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start(self) -> Session:
        session = Session()
        self._save(session)
        return session

    def end(self, session_id: str) -> Session | None:
        session = self.get(session_id)
        if session is None:
            return None
        session.status = "ended"
        session.updated_at = datetime.now(UTC)
        self._save(session)
        return session

    def get(self, session_id: str) -> Session | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def update(
        self,
        session_id: str,
        *,
        turn_count: int | None = None,
        summary: str | None = None,
        summarized_through: int | None = None,
    ) -> Session | None:
        session = self.get(session_id)
        if session is None:
            return None
        if turn_count is not None:
            session.turn_count = turn_count
        if summary is not None:
            session.summary = summary
        if summarized_through is not None:
            session.summarized_through = summarized_through
        session.updated_at = datetime.now(UTC)
        self._save(session)
        return session

    def _save(self, session: Session) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, created_at, updated_at, status, turn_count, summary, summarized_through)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session.id,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.status,
                session.turn_count,
                session.summary,
                session.summarized_through,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            turn_count=row["turn_count"],
            summary=row["summary"],
            summarized_through=row["summarized_through"],
        )
