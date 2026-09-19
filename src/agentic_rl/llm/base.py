from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_rl.core.models import Candidate, State, Step


class Planner(ABC):
    """Proposes candidate actions for a request. Does not choose between them —
    that's the Policy's job (see policy/base.py)."""

    id: str

    @abstractmethod
    async def plan(
        self,
        state: State,
        tool_schemas: list[dict],
        prior_corrections: list[str],
        rules: list[str] | None = None,
        history: list[Step] | None = None,
    ) -> list[Candidate]:
        """Return one or more candidate actions, most-likely-correct first.

        `prior_corrections` are short strings like "don't do X, do Y instead" retrieved
        from past episodes with a similar request (see core/store.py); they should be
        rendered into the prompt so mistakes aren't repeated on the very next attempt,
        ahead of the bandit converging.

        `rules` are consolidated, deduplicated standing preferences (core/models.py
        Memory, via core/memory.py MemoryStore.active_rules) — the user's binding
        instructions, distinct from `prior_corrections` which are episode-specific
        and only weakly related by keyword search.

        `history` is the steps already executed in this episode (core/agent.py's
        multi-step loop), empty/None on the first call. When non-empty, candidates
        should build on those steps' outcomes (e.g. use a prior payload as this step's
        params), never repeat a completed step, and propose the `answer` capability
        with the final user-facing text once nothing further is needed.
        """
        ...
