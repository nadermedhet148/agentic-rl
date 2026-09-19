from __future__ import annotations

import pytest

from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import Action, Candidate, Episode, Outcome, State
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.distiller import Distiller, DistillResult, MockDistiller


def make_episode(request: str = "fetch orders from https://api.example.com/orders") -> Episode:
    candidate = Candidate(capability="http_call", params={"method": "GET", "url": "https://api.example.com/orders"})
    action = Action(candidate=candidate, index=0, explored=False, arm_id="http_call:abc:False")
    return Episode(
        state=State(request=request),
        candidates=[candidate],
        action=action,
        outcome=Outcome(ok=True, status="200"),
        implicit_reward=0.2,
        planner_id="mock",
        policy_id="linucb",
    )


@pytest.fixture
def episode_store():
    s = EpisodeStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def memory_store(episode_store):
    return MemoryStore(episode_store.connection)


# --- MemoryStore --------------------------------------------------------------


def test_add_and_get(memory_store):
    memory = memory_store.add("always send Accept: application/json to api.example.com")
    fetched = memory_store.get(memory.id)
    assert fetched is not None
    assert fetched.text == memory.text
    assert fetched.support_count == 1
    assert fetched.active is True


def test_get_missing_returns_none(memory_store):
    assert memory_store.get("nope") is None


def test_bump_support_increments_and_tracks_source_episodes(memory_store):
    memory = memory_store.add("use json header", source_episode_ids=["ep-1"])
    bumped = memory_store.bump_support(memory.id, "ep-2")
    assert bumped.support_count == 2
    assert set(bumped.source_episode_ids) == {"ep-1", "ep-2"}


def test_bump_support_dedupes_episode_ids(memory_store):
    memory = memory_store.add("use json header", source_episode_ids=["ep-1"])
    bumped = memory_store.bump_support(memory.id, "ep-1")
    assert bumped.support_count == 2
    assert bumped.source_episode_ids == ["ep-1"]


def test_bump_support_missing_id_returns_none(memory_store):
    assert memory_store.bump_support("nope", "ep-1") is None


def test_supersede_deactivates_old_and_links_to_new(memory_store):
    old = memory_store.add("never confirm orders")
    new = memory_store.add("always confirm orders")
    result = memory_store.supersede(old.id, new.id)
    assert result.active is False
    assert result.superseded_by == new.id

    refetched = memory_store.get(old.id)
    assert refetched.active is False


def test_deactivate(memory_store):
    memory = memory_store.add("some rule")
    memory_store.deactivate(memory.id)
    assert memory_store.get(memory.id).active is False


def test_active_rules_excludes_inactive(memory_store):
    keep = memory_store.add("keep me")
    drop = memory_store.add("drop me")
    memory_store.deactivate(drop.id)

    rules = memory_store.active_rules()
    ids = {r.id for r in rules}
    assert keep.id in ids
    assert drop.id not in ids


def test_active_rules_ordered_by_support_then_recency(memory_store):
    low = memory_store.add("low support rule")
    high = memory_store.add("high support rule")
    memory_store.bump_support(high.id, "ep-1")
    memory_store.bump_support(high.id, "ep-2")  # support_count = 3

    rules = memory_store.active_rules()
    assert rules[0].id == high.id
    assert rules[1].id == low.id


def test_active_rules_filters_by_capability(memory_store):
    scoped = memory_store.add("http-only rule", capability="http_call")
    general = memory_store.add("general rule", capability=None)
    other = memory_store.add("schedule-only rule", capability="schedule_task")

    rules = memory_store.active_rules(capability="http_call")
    ids = {r.id for r in rules}
    assert scoped.id in ids
    assert general.id in ids
    assert other.id not in ids


def test_search_finds_overlapping_rule(memory_store):
    memory_store.add("always send Accept: application/json to api.example.com")
    results = memory_store.search("call the users endpoint on api.example.com")
    assert results
    assert "Accept" in results[0].text


def test_search_does_not_match_unrelated_text(memory_store):
    memory_store.add("always send Accept: application/json to api.example.com")
    results = memory_store.search("schedule a weather report")
    assert results == []


def test_search_excludes_inactive_rules(memory_store):
    memory = memory_store.add("always send Accept: application/json to api.example.com")
    memory_store.deactivate(memory.id)
    results = memory_store.search("call api.example.com")
    assert results == []


# --- Consolidator --------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_creates_new_rule(memory_store):
    consolidator = Consolidator(memory_store, MockDistiller())
    episode = make_episode()

    memory = await consolidator.consolidate("always include Content-Type: application/json", episode)

    assert memory.text == "always include Content-Type: application/json"
    assert memory.source_episode_ids == [episode.id]
    stored = memory_store.get(memory.id)
    assert stored is not None


@pytest.mark.asyncio
async def test_consolidate_same_correction_twice_bumps_support_once(memory_store):
    consolidator = Consolidator(memory_store, MockDistiller())
    correction = "always include Content-Type: application/json"

    first = await consolidator.consolidate(correction, make_episode())
    second = await consolidator.consolidate(correction, make_episode())

    assert first.id == second.id
    assert second.support_count == 2
    assert len(memory_store.active_rules()) == 1


@pytest.mark.asyncio
async def test_consolidate_ignores_hallucinated_match_id(memory_store):
    class FakeDistiller(Distiller):
        id = "fake"

        async def distill(self, correction, episode, existing) -> DistillResult:
            return DistillResult(rule_text=correction, matches_existing_id="not-a-real-id")

    consolidator = Consolidator(memory_store, FakeDistiller())
    memory = await consolidator.consolidate("some correction", make_episode())

    # the bogus id was ignored, so a brand-new rule was created instead of an error
    assert memory_store.get(memory.id) is not None
    assert memory.support_count == 1


@pytest.mark.asyncio
async def test_consolidate_supersedes_contradicted_rule(memory_store):
    old = memory_store.add("never confirm orders")

    class SupersedingDistiller(Distiller):
        id = "fake"

        async def distill(self, correction, episode, existing) -> DistillResult:
            return DistillResult(rule_text="always confirm orders", supersedes_id=old.id)

    consolidator = Consolidator(memory_store, SupersedingDistiller())
    new = await consolidator.consolidate("actually always confirm orders", make_episode())

    assert memory_store.get(old.id).active is False
    assert memory_store.get(old.id).superseded_by == new.id
    assert new.active is True


@pytest.mark.asyncio
async def test_consolidate_ignores_hallucinated_supersede_id(memory_store):
    class FakeDistiller(Distiller):
        id = "fake"

        async def distill(self, correction, episode, existing) -> DistillResult:
            return DistillResult(rule_text=correction, supersedes_id="not-a-real-id")

    consolidator = Consolidator(memory_store, FakeDistiller())
    memory = await consolidator.consolidate("some correction", make_episode())

    # nothing was superseded because the id didn't correspond to any known rule
    assert memory_store.get(memory.id) is not None
