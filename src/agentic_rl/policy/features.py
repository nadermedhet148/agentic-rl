from __future__ import annotations

import hashlib
import json

import numpy as np

from agentic_rl.core.models import Candidate, State

FEATURE_DIM = 32
_HASH_BUCKETS = FEATURE_DIM - 8  # first 8 dims are hand-picked; the rest is a hashed arm id


def _param_template(capability: str, params: dict) -> dict:
    """The *shape* of a candidate's params, not its values — so learning generalizes
    across URLs/instructions instead of memorizing one exact call. See policy/features.py
    docstring in docs/PLAN.md for the rationale."""
    if capability == "http_call":
        return {
            "method": str(params.get("method", "GET")).upper(),
            "header_keys": sorted((params.get("headers") or {}).keys()),
            "has_body": params.get("json_body") is not None,
        }
    if capability == "delegate":
        # which peer is the decision — so agents learn *whom* to ask, per peer
        return {"agent_id": str(params.get("agent_id", ""))}
    if capability == "schedule_task":
        return {
            "kind": "cron" if params.get("cron") else "run_at",
            "has_timezone": bool(params.get("timezone")),
        }
    return {"keys": sorted(params.keys())}


def param_template_hash(capability: str, params: dict) -> str:
    encoded = json.dumps(_param_template(capability, params), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]


def arm_id(candidate: Candidate) -> str:
    """Stable identity for a candidate as a bandit arm: same capability + same param
    shape + same confirmation requirement are treated as "the same decision" across
    different requests."""
    template_hash = param_template_hash(candidate.capability, candidate.params)
    return f"{candidate.capability}:{template_hash}:{candidate.needs_confirmation}"


def build_features(
    state: State,
    candidate: Candidate,
    capability_success_rate: float,
    correction_count: int,
) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[0] = 1.0  # bias
    vec[1] = state.hour_of_day / 23.0
    vec[2] = min(state.prior_correction_count, 10) / 10.0
    vec[3] = candidate.confidence
    vec[4] = 1.0 if candidate.needs_confirmation else 0.0
    vec[5] = capability_success_rate
    vec[6] = min(correction_count, 10) / 10.0
    vec[7] = 1.0 if state.source == "scheduler" else 0.0

    digest = hashlib.sha256(arm_id(candidate).encode()).digest()
    bucket = 8 + (int.from_bytes(digest[:8], "big") % _HASH_BUCKETS)
    vec[bucket] = 1.0
    return vec
