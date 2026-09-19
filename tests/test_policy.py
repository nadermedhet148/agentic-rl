from __future__ import annotations

import numpy as np
import pytest

from agentic_rl.core.models import Candidate, State
from agentic_rl.policy.base import Arm
from agentic_rl.policy.epsilon import EpsilonGreedyPolicy
from agentic_rl.policy.features import FEATURE_DIM, arm_id, build_features, param_template_hash
from agentic_rl.policy.greedy import GreedyPolicy
from agentic_rl.policy.linucb import LinUCBPolicy


def _synthetic_arms() -> tuple[list[Arm], np.ndarray]:
    """3 arms with fixed, distinguishable one-hot-ish features and known true means."""
    true_means = np.array([0.9, 0.5, 0.1])
    arms = []
    for i, mean in enumerate(true_means):
        features = np.zeros(FEATURE_DIM)
        features[0] = 1.0  # bias
        features[3 + i] = 1.0  # distinguishing feature per arm
        arms.append(Arm(id=f"arm-{i}", features=features, confidence=0.5))
    return arms, true_means


def _run_bandit(policy, n_rounds: int, rng: np.random.Generator) -> list[int]:
    arms, true_means = _synthetic_arms()
    chosen_history = []
    for _ in range(n_rounds):
        idx, _explored = policy.select(arms)
        chosen_history.append(idx)
        observed = float(true_means[idx] + rng.normal(0, 0.05))
        policy.update(arms[idx], observed)
    return chosen_history


# --- features -----------------------------------------------------------------


def test_arm_id_stable_across_param_values_same_shape():
    c1 = Candidate(capability="http_call", params={"method": "GET", "url": "https://a.com"})
    c2 = Candidate(capability="http_call", params={"method": "GET", "url": "https://b.com/x"})
    assert arm_id(c1) == arm_id(c2)


def test_arm_id_differs_across_shape():
    get_c = Candidate(capability="http_call", params={"method": "GET", "url": "https://a.com"})
    post_c = Candidate(capability="http_call", params={"method": "POST", "url": "https://a.com"})
    assert arm_id(get_c) != arm_id(post_c)


def test_param_template_hash_deterministic():
    params = {"method": "get", "url": "https://a.com"}
    assert param_template_hash("http_call", params) == param_template_hash("http_call", dict(params))


def test_build_features_shape_and_bounds():
    state = State(request="fetch https://x", prior_correction_count=3)
    candidate = Candidate(capability="http_call", params={"method": "GET", "url": "https://x"}, confidence=0.7)
    vec = build_features(state, candidate, capability_success_rate=0.8, correction_count=2)
    assert vec.shape == (FEATURE_DIM,)
    assert vec[3] == pytest.approx(0.7)
    assert vec[5] == pytest.approx(0.8)


# --- LinUCB ---------------------------------------------------------------


def test_linucb_converges_to_best_arm():
    policy = LinUCBPolicy(dim=FEATURE_DIM, alpha=0.5)
    rng = np.random.default_rng(42)
    history = _run_bandit(policy, n_rounds=300, rng=rng)

    last_50 = history[-50:]
    best_arm_rate = last_50.count(0) / len(last_50)
    assert best_arm_rate > 0.7, f"expected LinUCB to mostly pick the best arm late, got rate={best_arm_rate}"


def test_linucb_beats_random_selection_in_cumulative_reward():
    rng = np.random.default_rng(1)
    policy = LinUCBPolicy(dim=FEATURE_DIM, alpha=0.5)
    arms, true_means = _synthetic_arms()

    linucb_total = 0.0
    for _ in range(200):
        idx, _ = policy.select(arms)
        r = float(true_means[idx] + rng.normal(0, 0.05))
        policy.update(arms[idx], r)
        linucb_total += r

    # a policy that always picked uniformly at random would average true_means.mean()
    random_expected_total = 200 * true_means.mean()
    assert linucb_total > random_expected_total


def test_linucb_explore_mask_excludes_masked_arms():
    policy = LinUCBPolicy(dim=FEATURE_DIM, alpha=5.0)  # high alpha => bonus would dominate
    arms, _ = _synthetic_arms()
    # mask out the arm LinUCB has the strongest incentive to explore (arm 2, unseen)
    mask = [True, True, False]
    for _ in range(20):
        idx, _explored = policy.select(arms, explore_mask=mask)
        assert idx != 2
        policy.update(arms[idx], 0.0)


def test_linucb_select_requires_arms():
    policy = LinUCBPolicy()
    with pytest.raises(ValueError):
        policy.select([])


# --- EpsilonGreedy ----------------------------------------------------------


def test_epsilon_greedy_converges_to_best_arm():
    rng = np.random.default_rng(7)
    policy = EpsilonGreedyPolicy(epsilon=0.1, rng=rng)
    history = _run_bandit(policy, n_rounds=500, rng=np.random.default_rng(8))

    last_100 = history[-100:]
    best_arm_rate = last_100.count(0) / len(last_100)
    assert best_arm_rate > 0.7


def test_epsilon_greedy_respects_explore_mask():
    rng = np.random.default_rng(3)
    policy = EpsilonGreedyPolicy(epsilon=1.0, rng=rng)  # always "explore"
    arms, _ = _synthetic_arms()
    for _ in range(30):
        idx, explored = policy.select(arms, explore_mask=[True, False, False])
        assert idx == 0
        assert explored is True


# --- Greedy baseline ---------------------------------------------------------


def test_greedy_policy_picks_highest_confidence_and_never_learns():
    policy = GreedyPolicy()
    arms = [
        Arm(id="a", features=np.zeros(FEATURE_DIM), confidence=0.2),
        Arm(id="b", features=np.zeros(FEATURE_DIM), confidence=0.9),
        Arm(id="c", features=np.zeros(FEATURE_DIM), confidence=0.5),
    ]
    idx, explored = policy.select(arms)
    assert idx == 1
    assert explored is False

    policy.update(arms[1], reward=-10.0)  # should have no effect on future picks
    idx2, _ = policy.select(arms)
    assert idx2 == 1


# --- persistence -------------------------------------------------------------


def test_linucb_state_dict_round_trip():
    policy = LinUCBPolicy(dim=FEATURE_DIM, alpha=0.5)
    arms, true_means = _synthetic_arms()
    rng = np.random.default_rng(5)
    for _ in range(20):
        idx, _ = policy.select(arms)
        policy.update(arms[idx], float(true_means[idx] + rng.normal(0, 0.05)))

    state = policy.state_dict()
    restored = LinUCBPolicy(dim=FEATURE_DIM, alpha=0.5)
    restored.load_state(state)

    # same decision from the restored policy as from the original, for every arm
    for arm in arms:
        orig_idx, _ = policy.select([arm])
        restored_idx, _ = restored.select([arm])
        assert orig_idx == restored_idx
    assert restored.state_dict() == state


def test_linucb_load_state_empty_is_noop():
    policy = LinUCBPolicy()
    policy.load_state({})
    arms, _ = _synthetic_arms()
    idx, _explored = policy.select(arms)  # should behave like a fresh policy
    assert 0 <= idx < len(arms)


def test_epsilon_greedy_state_dict_round_trip():
    rng = np.random.default_rng(2)
    policy = EpsilonGreedyPolicy(epsilon=0.2, rng=rng)
    arms, true_means = _synthetic_arms()
    for _ in range(20):
        idx, _ = policy.select(arms)
        policy.update(arms[idx], float(true_means[idx]))

    state = policy.state_dict()
    restored = EpsilonGreedyPolicy(epsilon=0.2)
    restored.load_state(state)

    assert restored.state_dict() == state
    # greedy pick (epsilon=0 override via mask trick not needed — just compare means)
    assert restored._means == policy._means
    assert restored._counts == policy._counts


def test_greedy_state_dict_is_empty_and_load_state_is_noop():
    policy = GreedyPolicy()
    assert policy.state_dict() == {}
    policy.load_state({"anything": 1})  # must not raise
    arms, _ = _synthetic_arms()
    idx, _ = policy.select(arms)
    assert idx == int(np.argmax([a.confidence for a in arms]))


def test_state_dict_is_json_serializable():
    import json

    policy = LinUCBPolicy()
    arms, _ = _synthetic_arms()
    policy.update(arms[0], 1.0)
    json.dumps(policy.state_dict())  # must not raise
