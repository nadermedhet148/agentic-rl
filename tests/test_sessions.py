from __future__ import annotations

import pytest

from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore


@pytest.fixture
def store():
    s = EpisodeStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def sessions(store):
    return SessionStore(store.connection)


def test_start_creates_an_active_session(sessions):
    session = sessions.start()
    assert session.status == "active"
    assert session.turn_count == 0
    assert session.summary == ""
    assert session.summarized_through == 0


def test_get_missing_returns_none(sessions):
    assert sessions.get("nope") is None


def test_get_roundtrips_a_started_session(sessions):
    session = sessions.start()
    fetched = sessions.get(session.id)
    assert fetched is not None
    assert fetched.id == session.id
    assert fetched.status == "active"


def test_end_sets_status_ended(sessions):
    session = sessions.start()
    ended = sessions.end(session.id)
    assert ended is not None
    assert ended.status == "ended"
    assert sessions.get(session.id).status == "ended"


def test_end_missing_returns_none(sessions):
    assert sessions.end("nope") is None


def test_update_sets_only_given_fields(sessions):
    session = sessions.start()

    after_turn = sessions.update(session.id, turn_count=3)
    assert after_turn.turn_count == 3
    assert after_turn.summary == ""
    assert after_turn.summarized_through == 0

    after_summary = sessions.update(session.id, summary="the gist", summarized_through=3)
    assert after_summary.turn_count == 3
    assert after_summary.summary == "the gist"
    assert after_summary.summarized_through == 3


def test_update_missing_returns_none(sessions):
    assert sessions.update("nope", turn_count=1) is None


def test_sessions_share_the_episode_store_connection(store):
    # SessionStore must work against the same connection EpisodeStore uses, same
    # pattern as MemoryStore, so a :memory: database stays one database.
    sessions_a = SessionStore(store.connection)
    sessions_b = SessionStore(store.connection)
    session = sessions_a.start()
    assert sessions_b.get(session.id) is not None
