from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from agentic_rl.core import text as text_util
from agentic_rl.core.models import Episode, Memory
from agentic_rl.llm.distiller import Distiller

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    capability TEXT,
    support_count INTEGER NOT NULL,
    source_episode_ids TEXT NOT NULL,
    superseded_by TEXT,
    active INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_capability ON memories(capability);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(memory_id UNINDEXED, text);
"""

# bm25() weight per indexed column: `memories_fts` has a single indexed column
# (`text`), so this is just the default — kept as a constant for symmetry with
# EpisodeStore's _BM25_WEIGHTS and in case a second column is added later.
_BM25_WEIGHTS = (1.0,)


class MemoryStore:
    """SQLite-backed standing rules ("semantic memory") — see core/agent.py, which
    injects `active_rules()` into every plan, and Consolidator below, which is what
    turns corrections into these rules.

    Shares its connection with EpisodeStore (via EpisodeStore.connection) rather than
    opening its own, so a `:memory:` database used by both stays a single database.
    """

    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(
        self,
        text: str,
        *,
        capability: str | None = None,
        source_episode_ids: list[str] | None = None,
    ) -> Memory:
        memory = Memory(text=text, capability=capability, source_episode_ids=source_episode_ids or [])
        self._save(memory)
        return memory

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def bump_support(self, memory_id: str, episode_id: str) -> Memory | None:
        memory = self.get(memory_id)
        if memory is None:
            return None
        memory.support_count += 1
        if episode_id not in memory.source_episode_ids:
            memory.source_episode_ids.append(episode_id)
        memory.updated_at = datetime.now(UTC)
        self._save(memory)
        return memory

    def supersede(self, old_id: str, new_id: str) -> Memory | None:
        old = self.get(old_id)
        if old is None:
            return None
        old.active = False
        old.superseded_by = new_id
        old.updated_at = datetime.now(UTC)
        self._save(old)
        return old

    def deactivate(self, memory_id: str) -> Memory | None:
        memory = self.get(memory_id)
        if memory is None:
            return None
        memory.active = False
        memory.updated_at = datetime.now(UTC)
        self._save(memory)
        return memory

    def active_rules(self, limit: int = 50, capability: str | None = None) -> list[Memory]:
        """Active rules, most-supported and most-recently-updated first — this is
        what gets injected into every plan (see llm/prompts.py render_rules())."""
        if capability is None:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE active = 1 ORDER BY support_count DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM memories WHERE active = 1 AND (capability = ? OR capability IS NULL)
                   ORDER BY support_count DESC, updated_at DESC LIMIT ?""",
                (capability, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Active rules whose text overlaps `query` — same ranked-FTS + min-overlap
        approach as EpisodeStore.search_corrections (core/text.py)."""
        query_tokens = text_util.tokenize(query)
        fts_query = text_util.fts_query(query_tokens)
        if not fts_query:
            return []
        required_overlap = text_util.min_overlap(query_tokens)
        weights = ", ".join(str(w) for w in _BM25_WEIGHTS)
        rows = self._conn.execute(
            f"""SELECT m.* FROM memories_fts f
                JOIN memories m ON m.id = f.memory_id
                WHERE memories_fts MATCH ? AND m.active = 1
                ORDER BY bm25(memories_fts, {weights}) LIMIT ?""",
            (fts_query, max(limit * 4, 20)),
        ).fetchall()

        results = []
        for row in rows:
            if len(query_tokens & text_util.tokenize(row["text"])) >= required_overlap:
                results.append(self._row_to_memory(row))
            if len(results) >= limit:
                break
        return results

    def _save(self, memory: Memory) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, created_at, updated_at, kind, text, capability, support_count,
                source_episode_ids, superseded_by, active)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                memory.id,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.kind,
                memory.text,
                memory.capability,
                memory.support_count,
                json.dumps(memory.source_episode_ids),
                memory.superseded_by,
                int(memory.active),
            ),
        )
        self._conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.id,))
        self._conn.execute(
            "INSERT INTO memories_fts (memory_id, text) VALUES (?, ?)", (memory.id, memory.text)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            kind=row["kind"],
            text=row["text"],
            capability=row["capability"],
            support_count=row["support_count"],
            source_episode_ids=json.loads(row["source_episode_ids"]),
            superseded_by=row["superseded_by"],
            active=bool(row["active"]),
        )


class Consolidator:
    """Turns a correction into a standing rule: distill it, then either bump an
    existing equivalent rule, supersede a contradicted one, or add a new one."""

    def __init__(self, memory_store: MemoryStore, distiller: Distiller):
        self._memory = memory_store
        self._distiller = distiller

    async def consolidate(self, correction: str, episode: Episode) -> Memory:
        existing = self._memory.search(correction, limit=5)
        if episode.action.candidate.capability:
            existing += [
                m
                for m in self._memory.active_rules(limit=10, capability=episode.action.candidate.capability)
                if m.id not in {e.id for e in existing}
            ]

        result = await self._distiller.distill(correction, episode, existing)
        existing_ids = {m.id for m in existing}

        # An LLM can name an id that wasn't actually offered to it — guard against
        # bumping or superseding the wrong rule on a hallucinated match.
        matches_id = result.matches_existing_id if result.matches_existing_id in existing_ids else None
        supersedes_id = result.supersedes_id if result.supersedes_id in existing_ids else None

        if matches_id:
            bumped = self._memory.bump_support(matches_id, episode.id)
            if bumped is not None:
                return bumped

        new_memory = self._memory.add(
            result.rule_text,
            capability=result.capability,
            source_episode_ids=[episode.id],
        )
        if supersedes_id:
            self._memory.supersede(supersedes_id, new_memory.id)
        return new_memory
