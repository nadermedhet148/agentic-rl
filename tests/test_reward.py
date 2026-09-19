from __future__ import annotations

from agentic_rl.rl import reward


def test_implicit_reward_components():
    assert reward.implicit_reward(executed_ok=True) == reward.EXEC_OK
    assert reward.implicit_reward(executed_ok=False) == reward.EXEC_FAIL
    assert reward.implicit_reward(executed_ok=None) == 0.0
    assert reward.implicit_reward(executed_ok=True, reissued=True) == reward.EXEC_OK + reward.REISSUED
    assert reward.implicit_reward(cancelled=True) == reward.CANCELLED


def test_explicit_reward_correction_forces_negative():
    assert reward.explicit_reward(1, correction="do it differently") == -1.0
    assert reward.explicit_reward(1, correction=None) == 1.0
    assert reward.explicit_reward(-1, correction=None) == -1.0
    assert reward.explicit_reward(0, correction=None) == 0.0


def test_final_reward_prefers_explicit_over_implicit():
    assert reward.final_reward(explicit_score=1, correction=None, implicit=-0.5) == 1.0
    assert reward.final_reward(explicit_score=None, correction=None, implicit=-0.5) == -0.5


def test_weighted_reward_upweights_explicit():
    explicit = reward.weighted_reward(explicit_score=1, correction=None, implicit=0.2)
    implicit_only = reward.weighted_reward(explicit_score=None, correction=None, implicit=0.2)
    assert explicit == 1.0 * reward.EXPLICIT_WEIGHT
    assert implicit_only == 0.2 * reward.IMPLICIT_WEIGHT
    assert explicit > implicit_only
