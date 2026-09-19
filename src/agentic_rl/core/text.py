"""Shared text ranking helpers for SQLite FTS5 lookups — used by EpisodeStore
(episodes/corrections) and MemoryStore (rules), so both search the same way and a
fix to the ranking logic applies everywhere at once."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
HOST_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)

# Dropped before matching/ranking — request phrasing filler that would otherwise
# dominate the overlap between two otherwise-unrelated requests (e.g. "fetch" alone
# matching everything). Tuned for how requests are phrased, not general-purpose prose.
STOPWORDS = {
    "fetch", "get", "call", "please", "the", "a", "an", "from", "to", "for",
    "at", "with", "and", "on", "of", "run", "make", "do", "using", "into", "via",
}


def extract_hosts(text: str) -> list[str]:
    return [h.lower() for h in HOST_RE.findall(text)]


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if len(t) >= 2} - STOPWORDS


def fts_query(tokens: set[str]) -> str:
    if not tokens:
        return ""
    # Quote each token so URLs/punctuation can't break FTS5 query syntax; OR them so
    # any shared token counts as a match — recall over precision at the FTS layer,
    # a min-token-overlap filter run by the caller afterward tightens precision.
    return " OR ".join(f'"{t}"' for t in sorted(tokens))


def min_overlap(query_tokens: set[str]) -> int:
    """How many tokens two token sets must share to count as "similar" — a single
    shared token is enough only when the query itself is that short to begin with."""
    return 1 if len(query_tokens) < 2 else 2
