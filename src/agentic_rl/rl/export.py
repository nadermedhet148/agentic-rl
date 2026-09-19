from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from agentic_rl.core.models import Episode
from agentic_rl.rl.reward import final_reward


def episode_to_record(episode: Episode) -> dict:
    """One episode as a chosen-vs-rejected preference record — the shape a future
    DPO/GRPO fine-tuning pass would consume."""
    reward = (
        episode.final_reward
        if episode.final_reward is not None
        else final_reward(episode.explicit_score, episode.correction, episode.implicit_reward)
    )
    rejected = [c for i, c in enumerate(episode.candidates) if i != episode.action.index]
    return {
        "episode_id": episode.id,
        "prompt": episode.state.request,
        "source": episode.state.source,
        "chosen": episode.action.candidate.model_dump(),
        "rejected": [c.model_dump() for c in rejected],
        "reward": reward,
        "explored": episode.action.explored,
        "correction": episode.correction,
    }


def export_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    """Write episodes as JSONL to `path`; returns the number of records written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode_to_record(episode)) + "\n")
            count += 1
    return count
