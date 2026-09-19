from __future__ import annotations

# Implicit reward components (see docs/PLAN.md rl/reward.py):
EXEC_OK = 0.2
EXEC_FAIL = -0.5
REISSUED = -0.3
CANCELLED = -0.5

# Explicit feedback is weighted this much more heavily than implicit-only signal when
# updating the policy, since it's a deliberate, low-noise signal from the user.
EXPLICIT_WEIGHT = 3.0
IMPLICIT_WEIGHT = 1.0


def implicit_reward(
    *,
    executed_ok: bool | None = None,
    reissued: bool = False,
    cancelled: bool = False,
) -> float:
    """Reward inferred from what happened, without the user saying anything.
    Deliberately capped/small relative to explicit feedback — see EXPLICIT_WEIGHT."""
    total = 0.0
    if executed_ok is True:
        total += EXEC_OK
    elif executed_ok is False:
        total += EXEC_FAIL
    if reissued:
        total += REISSUED
    if cancelled:
        total += CANCELLED
    return total


def explicit_reward(score: int, correction: str | None) -> float:
    """A correction always means the action was wrong, regardless of the raw score."""
    if correction:
        return -1.0
    return float(score)


def final_reward(
    explicit_score: int | None,
    correction: str | None,
    implicit: float,
) -> float:
    """The reward stored on the episode: explicit feedback if the user gave any,
    otherwise whatever was inferred implicitly."""
    if explicit_score is not None:
        return explicit_reward(explicit_score, correction)
    return implicit


def update_weight(explicit_score: int | None) -> float:
    """How much to weight `final_reward` when calling Policy.update — see EXPLICIT_WEIGHT."""
    return EXPLICIT_WEIGHT if explicit_score is not None else IMPLICIT_WEIGHT


def weighted_reward(explicit_score: int | None, correction: str | None, implicit: float) -> float:
    """final_reward(...) scaled by update_weight(...) — what Policy.update should be called with."""
    return final_reward(explicit_score, correction, implicit) * update_weight(explicit_score)
