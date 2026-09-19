from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from agentic_rl.core.models import Episode
from agentic_rl.rl.reward import final_reward


def episode_to_records(episode: Episode) -> list[dict]:
    """One record per step, each a chosen-vs-rejected preference pair — the shape a
    future DPO/GRPO fine-tuning pass would consume. All steps of an episode share its
    episode-level reward (feedback is recorded per episode, not per step)."""
    reward = (
        episode.final_reward
        if episode.final_reward is not None
        else final_reward(episode.explicit_score, episode.correction, episode.implicit_reward)
    )
    records = []
    for step in episode.steps:
        rejected = [c for i, c in enumerate(step.candidates) if i != step.action.index]
        records.append(
            {
                "episode_id": episode.id,
                "step_index": step.index,
                "prompt": episode.state.request,
                "source": episode.state.source,
                "chosen": step.action.candidate.model_dump(),
                "rejected": [c.model_dump() for c in rejected],
                "reward": reward,
                "explored": step.action.explored,
                "correction": episode.correction,
                "answer": episode.answer,
            }
        )
    return records


def export_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    """Write one JSONL record per step across all episodes to `path`; returns the
    number of records written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for episode in episodes:
            for record in episode_to_records(episode):
                f.write(json.dumps(record) + "\n")
                count += 1
    return count
