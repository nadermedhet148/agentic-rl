from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agentic_rl.policy.base import Arm, PeerEvidence, Policy

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_trust (
    agent_id TEXT NOT NULL,
    peer_id TEXT NOT NULL,
    key TEXT NOT NULL,
    err_peer REAL NOT NULL,
    err_self REAL NOT NULL,
    n INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, peer_id, key)
);
"""

PAIR_KEY = "*"  # trust key for the agent pair as a whole, across every arm
_EPS = 0.05  # keeps the trust ratio defined when both errors are ~0


@dataclass
class _TrustStat:
    err_peer: float  # EMA of (reward - peer's prediction)^2 on this agent's rewards
    err_self: float  # EMA of (reward - this agent's own prediction)^2
    n: int = 0


@dataclass
class _Member:
    policy: Policy
    share: bool
    synced_epoch: int = -1


class KnowledgeHub:
    """How agents in a team learn from each other — see docs/MULTI-AGENT-PLAN.md.

    Procedural sharing: each agent's policy scores with its own evidence plus every
    peer's `Policy.evidence()`, weighted by trust (`refresh`). Evidence is recomputed
    from peers' *local* stats and replaced each time, never accumulated, so pooling
    can't double-count.

    Trust: w[i][j] — how much agent i should believe agent j — is learned online,
    per arm and per pair: every time i observes a reward on an arm, j's own
    prediction for that arm is scored against it, and so is i's (`observe`). A peer
    that predicts i's rewards about as well as i itself does (same user, same
    preferences) is trusted fully; one that keeps predicting the opposite (a
    conflicting preference) drifts toward 0 on exactly the arms where they disagree.

    Everything that crosses between agents is `Policy.evidence()` — plain JSON —
    so moving agents into separate processes later means shipping those dicts over
    a transport instead of reading them from `self._members`; the math is unchanged.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        prior: float = 0.5,
        beta: float = 0.1,
        min_obs: int = 2,
        enabled: bool = True,
    ):
        self._conn = connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._prior = prior
        self._beta = beta
        self._min_obs = min_obs
        self.enabled = enabled
        self._members: dict[str, _Member] = {}
        self._trust: dict[tuple[str, str, str], _TrustStat] = {}
        self._epoch = 0
        for row in self._conn.execute("SELECT * FROM agent_trust").fetchall():
            self._trust[(row[0], row[1], row[2])] = _TrustStat(row[3], row[4], row[5])

    # --- membership ----------------------------------------------------------------

    def register(self, agent_id: str, policy: Policy, share: bool = True) -> None:
        self._members[agent_id] = _Member(policy=policy, share=share)
        self._epoch += 1

    def agent_ids(self) -> list[str]:
        return list(self._members)

    def _peers(self, agent_id: str) -> list[str]:
        """Peers whose knowledge `agent_id` may use: sharing must be on for the hub
        and for both ends of the pair."""
        me = self._members.get(agent_id)
        if not self.enabled or me is None or not me.share:
            return []
        return [pid for pid, m in self._members.items() if pid != agent_id and m.share]

    # --- trust ---------------------------------------------------------------------

    def trust(self, agent_id: str, peer_id: str, arm_id: str | None = None) -> float:
        """w[agent][peer] for one arm (falling back to the pair-level estimate while
        that arm has too few observations), or for the pair as a whole."""
        if arm_id is not None:
            stat = self._trust.get((agent_id, peer_id, arm_id))
            if stat is not None and stat.n >= self._min_obs:
                return self._weight(stat)
        stat = self._trust.get((agent_id, peer_id, PAIR_KEY))
        if stat is not None and stat.n >= self._min_obs:
            return self._weight(stat)
        return self._prior

    @staticmethod
    def _weight(stat: _TrustStat) -> float:
        # 1.0 when the peer predicts this agent's rewards at least as well as its own
        # model does; falls off quadratically as the peer's error dominates. Squared
        # because a peer usually has far more evidence on an arm than this agent does:
        # a linear fall-off left enough of a conflicting peer's weight to outvote the
        # agent's own early evidence (tuned on sim/team.py's conflict scenario).
        ratio = (stat.err_self + _EPS) / (stat.err_peer + _EPS)
        return max(0.0, min(1.0, ratio * ratio))

    def peer_weights(self, agent_id: str) -> dict[str, float]:
        """Pair-level trust in every shareable peer — the single number that also
        ranks peers' rules, corrections and demonstrations (core/agent.py)."""
        return {pid: self.trust(agent_id, pid) for pid in self._peers(agent_id)}

    def trust_matrix(self) -> list[dict[str, Any]]:
        rows = []
        for agent_id in self._members:
            for peer_id in self._members:
                if peer_id == agent_id:
                    continue
                stat = self._trust.get((agent_id, peer_id, PAIR_KEY))
                rows.append(
                    {
                        "agent_id": agent_id,
                        "peer_id": peer_id,
                        "trust": self.trust(agent_id, peer_id),
                        "observations": stat.n if stat else 0,
                        "sharing": peer_id in self._peers(agent_id),
                    }
                )
        return rows

    def observe(self, agent_id: str, arm: Arm, reward: float) -> None:
        """Score every peer's prediction (and this agent's own) against a reward this
        agent just observed. Call *before* the agent's own `policy.update`, so its
        self-prediction is an honest out-of-sample one."""
        me = self._members.get(agent_id)
        if me is None:
            return
        own = me.policy.predict(arm)
        err_self = (reward - (own if own is not None else 0.0)) ** 2
        for peer_id, peer in self._members.items():
            if peer_id == agent_id:
                continue
            predicted = peer.policy.predict(arm)
            if predicted is None:
                continue  # the peer knows nothing about this arm: no signal either way
            err_peer = (reward - predicted) ** 2
            for key in (arm.id, PAIR_KEY):
                self._update_stat(agent_id, peer_id, key, err_peer, err_self)
        self._epoch += 1

    def _update_stat(self, agent_id: str, peer_id: str, key: str, err_peer: float, err_self: float) -> None:
        stat = self._trust.get((agent_id, peer_id, key))
        if stat is None:
            stat = _TrustStat(err_peer=err_peer, err_self=err_self, n=1)
        else:
            stat.err_peer += self._beta * (err_peer - stat.err_peer)
            stat.err_self += self._beta * (err_self - stat.err_self)
            stat.n += 1
        self._trust[(agent_id, peer_id, key)] = stat
        self._conn.execute(
            """INSERT OR REPLACE INTO agent_trust
               (agent_id, peer_id, key, err_peer, err_self, n, updated_at) VALUES (?,?,?,?,?,?,?)""",
            (agent_id, peer_id, key, stat.err_peer, stat.err_self, stat.n, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    # --- procedural pooling ----------------------------------------------------------

    def mark_updated(self, agent_id: str) -> None:
        """An agent's local evidence changed — peers re-pool on their next refresh."""
        self._epoch += 1

    def refresh(self, agent_id: str) -> None:
        """Hand `agent_id`'s policy the current trust-weighted evidence of its peers.
        Lazy: a no-op unless some agent's evidence or trust changed since last time."""
        me = self._members.get(agent_id)
        if me is None or me.synced_epoch == self._epoch:
            return
        peers = [
            PeerEvidence(
                agent_id=peer_id,
                evidence=self._members[peer_id].policy.evidence(),
                weight=lambda arm_id, peer_id=peer_id: self.trust(agent_id, peer_id, arm_id),
            )
            for peer_id in self._peers(agent_id)
        ]
        me.policy.set_peer_evidence(peers)
        me.synced_epoch = self._epoch
