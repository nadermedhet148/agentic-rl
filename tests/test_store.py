from __future__ import annotations

import sqlite3

import pytest

from agentic_rl.core.models import Action, Candidate, Episode, Feedback, Outcome, State
from agentic_rl.core.store import EpisodeStore
from agentic_rl.rl.export import episode_to_record, export_jsonl


def make_episode(
    request: str = "fetch https://api.example.com/thing",
    capability: str = "http_call",
    outcome_ok: bool | None = True,
    correction: str | None = None,
    explicit_score: int | None = None,
) -> Episode:
    candidate = Candidate(capability=capability, params={"method": "GET", "url": "https://x"}, confidence=0.8)
    action = Action(candidate=candidate, index=0, explored=False, arm_id=f"{capability}:abc:False")
    outcome = None if outcome_ok is None else Outcome(ok=outcome_ok, status="200" if outcome_ok else "500")
    return Episode(
        state=State(request=request),
        candidates=[candidate],
        action=action,
        outcome=outcome,
        implicit_reward=0.2 if outcome_ok else -0.5,
        correction=correction,
        explicit_score=explicit_score,
        planner_id="mock",
        policy_id="linucb",
    )


@pytest.fixture
def store():
    s = EpisodeStore(":memory:")
    yield s
    s.close()


def test_save_and_get_roundtrip(store):
    episode = make_episode()
    store.save(episode)
    fetched = store.get(episode.id)
    assert fetched is not None
    assert fetched.id == episode.id
    assert fetched.state.request == episode.state.request


def test_get_missing_returns_none(store):
    assert store.get("does-not-exist") is None


def test_apply_feedback_sets_final_reward_and_persists(store):
    episode = make_episode(outcome_ok=True)
    store.save(episode)

    updated = store.apply_feedback(Feedback(episode_id=episode.id, score=1, correction=None))
    assert updated.final_reward == 1.0

    refetched = store.get(episode.id)
    assert refetched.final_reward == 1.0
    assert refetched.explicit_score == 1


def test_apply_feedback_correction_forces_negative_final_reward(store):
    episode = make_episode(outcome_ok=True)
    store.save(episode)

    updated = store.apply_feedback(
        Feedback(episode_id=episode.id, score=1, correction="always confirm deletes")
    )
    assert updated.final_reward == -1.0
    assert updated.correction == "always confirm deletes"


def test_apply_feedback_unknown_episode_raises(store):
    with pytest.raises(KeyError):
        store.apply_feedback(Feedback(episode_id="nope", score=1))


def test_search_corrections_finds_similar_request(store):
    episode = make_episode(
        request="fetch https://api.example.com/orders",
        correction="always include Content-Type: application/json",
    )
    store.save(episode)

    results = store.search_corrections("please fetch orders from example.com")
    assert any("Content-Type" in r for r in results)


def test_search_corrections_ignores_episodes_without_correction(store):
    store.save(make_episode(request="fetch https://api.example.com/orders", correction=None))
    results = store.search_corrections("fetch orders")
    assert results == []


def test_search_corrections_empty_query_returns_empty(store):
    assert store.search_corrections("") == []


def test_correction_count(store):
    store.save(make_episode(request="fetch orders", correction="use json"))
    store.save(make_episode(request="fetch orders again", correction="use json again"))
    assert store.correction_count("fetch orders") == 2


def test_capability_success_rate(store):
    store.save(make_episode(capability="http_call", outcome_ok=True))
    store.save(make_episode(capability="http_call", outcome_ok=True))
    store.save(make_episode(capability="http_call", outcome_ok=False))
    rate = store.capability_success_rate("http_call")
    assert rate == pytest.approx(2 / 3)


def test_capability_success_rate_default_when_no_data(store):
    assert store.capability_success_rate("schedule_task") == 0.5


def test_rolling_reward_chronological_order(store):
    import time

    for i in range(3):
        ep = make_episode(outcome_ok=True)
        store.save(ep)
        time.sleep(0.001)

    rewards = store.rolling_reward(n=3)
    assert len(rewards) == 3
    assert all(r == pytest.approx(0.2) for r in rewards)


def test_list_episodes_respects_limit(store):
    for _ in range(5):
        store.save(make_episode())
    assert len(store.list_episodes(limit=2)) == 2
    assert len(store.list_episodes(limit=10)) == 5


def test_save_is_upsert(store):
    episode = make_episode()
    store.save(episode)
    episode.explicit_score = 1
    episode.final_reward = 1.0
    store.save(episode)

    assert len(store.list_episodes(limit=10)) == 1
    assert store.get(episode.id).final_reward == 1.0


# --- export -----------------------------------------------------------------


def test_episode_to_record_shapes_chosen_and_rejected():
    alt = Candidate(capability="http_call", params={"method": "POST", "url": "https://x"})
    episode = make_episode()
    episode.candidates = [episode.action.candidate, alt]
    record = episode_to_record(episode)
    assert record["chosen"]["capability"] == "http_call"
    assert len(record["rejected"]) == 1
    assert record["reward"] == pytest.approx(0.2)


def test_export_jsonl_writes_one_line_per_episode(tmp_path, store):
    store.save(make_episode())
    store.save(make_episode())
    episodes = store.list_episodes(limit=10)

    out = tmp_path / "episodes.jsonl"
    count = export_jsonl(episodes, out)

    assert count == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


# --- policy state -----------------------------------------------------------


def test_save_and_load_policy_state(store):
    assert store.load_policy_state("linucb") is None

    store.save_policy_state("linucb", {"dim": 32, "arms": {"a": {"A": [[1]], "b": [1]}}})
    loaded = store.load_policy_state("linucb")
    assert loaded == {"dim": 32, "arms": {"a": {"A": [[1]], "b": [1]}}}


def test_save_policy_state_is_upsert(store):
    store.save_policy_state("linucb", {"v": 1})
    store.save_policy_state("linucb", {"v": 2})
    assert store.load_policy_state("linucb") == {"v": 2}


def test_policy_state_is_scoped_by_policy_id(store):
    store.save_policy_state("linucb", {"v": "linucb-state"})
    store.save_policy_state("epsilon", {"v": "epsilon-state"})
    assert store.load_policy_state("linucb") == {"v": "linucb-state"}
    assert store.load_policy_state("epsilon") == {"v": "epsilon-state"}


# --- ranked retrieval (stopwords, host boost, min-overlap) -------------------


def test_search_corrections_matches_across_different_wording_via_shared_host(store):
    store.save(
        make_episode(
            request="fetch orders from https://api.example.com/orders",
            correction="always include Content-Type: application/json",
        )
    )
    results = store.search_corrections("fetch users from https://api.example.com/users")
    assert any("Content-Type" in r for r in results)


def test_search_corrections_does_not_match_unrelated_request(store):
    store.save(
        make_episode(
            request="fetch orders from https://api.example.com/orders",
            correction="always include Content-Type: application/json",
        )
    )
    results = store.search_corrections("fetch the weather forecast for tomorrow")
    assert results == []


def test_search_corrections_rejects_single_stopword_overlap(store):
    # "fetch" is a stopword and shouldn't alone justify a match between two
    # otherwise-unrelated requests.
    store.save(make_episode(request="fetch orders from https://api.example.com", correction="use json"))
    results = store.search_corrections("fetch a completely different report")
    assert results == []


def test_search_corrections_ranks_host_match_above_plain_word_match(store):
    store.save(
        make_episode(
            request="fetch data from https://api.example.com/data",
            correction="host-specific correction",
        )
    )
    store.save(
        make_episode(
            request="fetch some other data report thing",
            correction="generic correction",
        )
    )
    results = store.search_corrections("fetch data from https://api.example.com/other")
    assert results
    assert "host-specific correction" in results[0]


def test_fts_rebuilds_hosts_column_from_pre_existing_episodes(tmp_path):
    db_path = tmp_path / "old.db"

    # simulate a pre-existing DB created before the `hosts` column existed
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, request TEXT NOT NULL,
            source TEXT NOT NULL, capability TEXT NOT NULL, arm_id TEXT NOT NULL,
            status TEXT NOT NULL, outcome_ok INTEGER, implicit_reward REAL NOT NULL,
            explicit_score INTEGER, correction TEXT, final_reward REAL,
            planner_id TEXT NOT NULL, policy_id TEXT NOT NULL, job_id TEXT, data TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE episodes_fts USING fts5(episode_id UNINDEXED, request, correction);
        """
    )
    old_episode = make_episode(
        request="fetch orders from https://api.example.com/orders",
        correction="always include Content-Type: application/json",
    )
    conn.execute(
        """INSERT INTO episodes (id, created_at, request, source, capability, arm_id, status,
           outcome_ok, implicit_reward, explicit_score, correction, final_reward, planner_id,
           policy_id, job_id, data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            old_episode.id, old_episode.created_at.isoformat(), old_episode.state.request,
            old_episode.state.source, old_episode.action.candidate.capability,
            old_episode.action.arm_id, old_episode.status, 1, old_episode.implicit_reward,
            old_episode.explicit_score, old_episode.correction, old_episode.final_reward,
            old_episode.planner_id, old_episode.policy_id, None, old_episode.model_dump_json(),
        ),
    )
    conn.execute(
        "INSERT INTO episodes_fts (episode_id, request, correction) VALUES (?, ?, ?)",
        (old_episode.id, old_episode.state.request, old_episode.correction),
    )
    conn.commit()
    conn.close()

    # opening with EpisodeStore should detect the missing `hosts` column and rebuild
    reopened = EpisodeStore(db_path)
    try:
        cols = {row["name"] for row in reopened.connection.execute("PRAGMA table_info(episodes_fts)").fetchall()}
        assert "hosts" in cols

        results = reopened.search_corrections("fetch users from https://api.example.com/users")
        assert any("Content-Type" in r for r in results)

        # the underlying episode row itself survived the rebuild untouched
        assert reopened.get(old_episode.id) is not None
    finally:
        reopened.close()
