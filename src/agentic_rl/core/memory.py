from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from agentic_rl.core import text as text_util
from agentic_rl.core.models import DEFAULT_AGENT_ID, Episode, Memory
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
    active INTEGER NOT NULL,
    owner_agent_id TEXT NOT NULL DEFAULT 'default',
    scope TEXT NOT NULL DEFAULT 'team',
    support_by_agent TEXT NOT NULL DEFAULT '{}',
    loosens_safety INTEGER NOT NULL DEFAULT 0,
    overrides_id TEXT
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
        self._ensure_agent_columns()

    def _ensure_agent_columns(self) -> None:
        """Dev-project migration for multi-agent support (docs/MULTI-AGENT-PLAN.md):
        rules created before it belong to the default agent and stay team-scoped,
        so every agent keeps seeing them — the pre-multi-agent behaviour."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "owner_agent_id" not in columns:
            self._conn.execute(
                f"ALTER TABLE memories ADD COLUMN owner_agent_id TEXT NOT NULL DEFAULT '{DEFAULT_AGENT_ID}'"
            )
        if "scope" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'team'")
        if "support_by_agent" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN support_by_agent TEXT NOT NULL DEFAULT '{}'")
        if "loosens_safety" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN loosens_safety INTEGER NOT NULL DEFAULT 0")
        if "overrides_id" not in columns:
            self._conn.execute("ALTER TABLE memories ADD COLUMN overrides_id TEXT")
        self._conn.commit()

    def add(
        self,
        text: str,
        *,
        capability: str | None = None,
        source_episode_ids: list[str] | None = None,
        owner_agent_id: str = DEFAULT_AGENT_ID,
        scope: str = "team",
        loosens_safety: bool = False,
        overrides_id: str | None = None,
    ) -> Memory:
        memory = Memory(
            text=text,
            capability=capability,
            source_episode_ids=source_episode_ids or [],
            owner_agent_id=owner_agent_id,
            scope=scope,
            support_by_agent={owner_agent_id: 1},
            loosens_safety=loosens_safety,
            overrides_id=overrides_id,
        )
        self._save(memory)
        return memory

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def bump_support(self, memory_id: str, episode_id: str, agent_id: str | None = None) -> Memory | None:
        memory = self.get(memory_id)
        if memory is None:
            return None
        memory.support_count += 1
        if agent_id is not None:
            memory.support_by_agent[agent_id] = memory.support_by_agent.get(agent_id, 0) + 1
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

    def promote(self, memory_id: str) -> Memory | None:
        """Make a private rule team-wide. If it was overriding a team rule for its
        owner, it now replaces that rule for everyone."""
        memory = self.get(memory_id)
        if memory is None:
            return None
        memory.scope = "team"
        memory.updated_at = datetime.now(UTC)
        self._save(memory)
        if memory.overrides_id:
            self.supersede(memory.overrides_id, memory.id)
        return memory

    def demote(self, memory_id: str) -> Memory | None:
        """Make a team rule private to its owner again."""
        memory = self.get(memory_id)
        if memory is None:
            return None
        memory.scope = "private"
        memory.updated_at = datetime.now(UTC)
        self._save(memory)
        return memory

    def deactivate(self, memory_id: str) -> Memory | None:
        memory = self.get(memory_id)
        if memory is None:
            return None
        memory.active = False
        memory.updated_at = datetime.now(UTC)
        self._save(memory)
        return memory

    def active_rules(
        self,
        limit: int = 50,
        capability: str | None = None,
        agent_id: str | None = None,
        capabilities: list[str] | None = None,
    ) -> list[Memory]:
        """Active rules, most-supported and most-recently-updated first — this is
        what gets injected into every plan (see llm/prompts.py render_rules()).

        With `agent_id`, only the rules that agent may see: every team-scoped rule
        plus its own private ones, minus any team rule one of its own private rules
        overrides (docs/MULTI-AGENT-PLAN.md, semantic sharing). With `capabilities`,
        capability-scoped rules are limited to those capabilities."""
        clauses = ["active = 1"]
        params: list = []
        if capability is not None:
            clauses.append("(capability = ? OR capability IS NULL)")
            params.append(capability)
        if capabilities is not None:
            placeholders = ", ".join("?" for _ in capabilities) or "NULL"
            clauses.append(f"(capability IS NULL OR capability IN ({placeholders}))")
            params.extend(capabilities)
        if agent_id is not None:
            clauses.append("(scope = 'team' OR owner_agent_id = ?)")
            clauses.append(
                "id NOT IN (SELECT overrides_id FROM memories WHERE active = 1 AND scope = 'private' "
                "AND owner_agent_id = ? AND overrides_id IS NOT NULL)"
            )
            params.extend([agent_id, agent_id])
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
            "ORDER BY support_count DESC, updated_at DESC LIMIT ?",
            (*params, limit),
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
                source_episode_ids, superseded_by, active, owner_agent_id, scope,
                support_by_agent, loosens_safety, overrides_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                memory.owner_agent_id,
                memory.scope,
                json.dumps(memory.support_by_agent),
                int(memory.loosens_safety),
                memory.overrides_id,
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
            owner_agent_id=row["owner_agent_id"],
            scope=row["scope"],
            support_by_agent=json.loads(row["support_by_agent"]),
            loosens_safety=bool(row["loosens_safety"]),
            overrides_id=row["overrides_id"],
        )


class Consolidator:
    """Turns a correction into a standing rule: distill it, then either bump an
    existing equivalent rule, supersede a contradicted one, or add a new one.

    In a team (docs/MULTI-AGENT-PLAN.md, semantic sharing), new rules start
    `default_scope` — "private" to the agent whose feedback produced them — and are
    promoted to "team" once a *second* agent's feedback independently matches them.
    A rule that loosens safety (skips a confirmation) is never auto-promoted; only a
    human can share it (MemoryStore.promote via the API).
    """

    def __init__(self, memory_store: MemoryStore, distiller: Distiller, default_scope: str = "team"):
        self._memory = memory_store
        self._distiller = distiller
        self._default_scope = default_scope

    async def consolidate(self, correction: str, episode: Episode) -> Memory:
        agent_id = episode.agent_id
        # search every agent's rules, not just this agent's — that's how a second
        # agent's matching correction gets counted as support for a peer's rule
        existing = self._memory.search(correction, limit=5)
        capabilities = {
            s.action.candidate.capability for s in episode.steps if s.action.candidate.capability != "answer"
        }
        capabilities.discard("")
        for capability in capabilities:
            existing += [
                m
                for m in self._memory.active_rules(limit=10, capability=capability)
                if m.id not in {e.id for e in existing}
            ]

        result = await self._distiller.distill(correction, episode, existing)
        by_id = {m.id: m for m in existing}

        # An LLM can name an id that wasn't actually offered to it — guard against
        # bumping or superseding the wrong rule on a hallucinated match.
        matches_id = result.matches_existing_id if result.matches_existing_id in by_id else None
        supersedes_id = result.supersedes_id if result.supersedes_id in by_id else None

        if matches_id:
            bumped = self._memory.bump_support(matches_id, episode.id, agent_id=agent_id)
            if bumped is not None:
                independent = [a for a, n in bumped.support_by_agent.items() if n > 0]
                if bumped.scope == "private" and len(independent) >= 2 and not bumped.loosens_safety:
                    return self._memory.promote(bumped.id) or bumped
                return bumped

        target = by_id.get(supersedes_id) if supersedes_id else None
        overrides_id = None
        if target is not None and self._default_scope == "private" and target.scope == "team":
            # A private rule may override a team rule for its owner only; it can't
            # retire the team rule for everyone (that takes a promotion).
            overrides_id, supersedes_id = supersedes_id, None
        elif target is not None and target.scope == "private" and target.owner_agent_id != agent_id:
            supersedes_id = None  # never retire another agent's private rule
        new_memory = self._memory.add(
            result.rule_text,
            capability=result.capability,
            source_episode_ids=[episode.id],
            owner_agent_id=agent_id,
            scope=self._default_scope,
            loosens_safety=result.loosens_safety,
            overrides_id=overrides_id,
        )
        if supersedes_id:
            self._memory.supersede(supersedes_id, new_memory.id)
        return new_memory
