from __future__ import annotations

import statistics

import pytest

from agentic_rl.sim.run import run_simulation


@pytest.mark.asyncio
async def test_linucb_learns_to_beat_its_own_early_performance():
    rewards, _memory = await run_simulation("linucb", episodes=300, seed=0)
    first = rewards[:100]
    last = rewards[-100:]
    assert statistics.mean(last) > statistics.mean(first)


@pytest.mark.asyncio
async def test_greedy_baseline_stays_flat_and_worse_than_linucb():
    linucb_rewards, _ = await run_simulation("linucb", episodes=300, seed=0)
    greedy_rewards, _ = await run_simulation("greedy", episodes=300, seed=0)

    linucb_last = statistics.mean(linucb_rewards[-100:])
    greedy_last = statistics.mean(greedy_rewards[-100:])
    assert linucb_last > greedy_last

    # greedy never learns: its first-100 and last-100 means should be indistinguishable
    greedy_first = statistics.mean(greedy_rewards[:100])
    assert abs(greedy_last - greedy_first) < 1e-9


@pytest.mark.asyncio
async def test_epsilon_greedy_also_learns():
    # This problem is easy enough (2 arms per request template) that epsilon-greedy
    # saturates well within the first 100 episodes, so a first-100-vs-last-100
    # comparison is too close to call — compare against the very first exposures
    # (before it's seen each arm more than once or twice) instead.
    rewards, _memory = await run_simulation("epsilon", episodes=400, seed=1)
    cold_start = rewards[:9]  # first 3 rounds through all 3 request templates
    settled = rewards[-100:]
    assert statistics.mean(settled) > statistics.mean(cold_start)
    assert statistics.mean(settled) > 0.7


@pytest.mark.asyncio
async def test_memory_consolidates_one_rule_per_scripted_preference():
    # 3 request templates each carry one scripted preference (sim/user.py) whose
    # correction text is identical every time the bad candidate is picked, so this
    # should converge on exactly 3 rules — not one per correction — regardless of
    # how many episodes ran (LinUCB here converges fast enough that each rule is
    # often only bumped once before the bad arm stops being picked at all).
    _rewards, memory = await run_simulation("linucb", episodes=300, seed=0)
    rules = memory.active_rules()
    assert len(rules) == 3
    assert all(rule.support_count >= 1 for rule in rules)


@pytest.mark.asyncio
async def test_memory_bumps_support_when_a_mistake_recurs():
    # epsilon-greedy keeps exploring (epsilon=0.1 by default) even after it's found
    # the better arm, so the bad arm — and its correction — recurs enough times
    # across 400 episodes to bump support past 1 on at least one rule.
    _rewards, memory = await run_simulation("epsilon", episodes=400, seed=1)
    rules = memory.active_rules()
    assert len(rules) == 3
    assert any(rule.support_count > 1 for rule in rules)
