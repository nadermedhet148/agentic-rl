from __future__ import annotations

from typing import Any

from agentic_rl.capabilities.base import Capability, Outcome, Tier

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The final answer to give the user."},
    },
    "required": ["text"],
    "additionalProperties": False,
}


class AnswerCapability(Capability):
    """Terminal step: reply to the user with the final answer text.

    Not a tool that touches the outside world — it's how the planner signals the
    agent loop (core/agent.py) is done. Registered on every Agent so it's always
    available to propose, and competes as an ordinary bandit arm so the policy
    learns when stopping is the right call.
    """

    name = "answer"
    description = (
        "Finish the request by replying to the user with the final answer text. "
        "Use only when no further capability calls are needed."
    )
    input_schema = INPUT_SCHEMA

    def tier_for(self, params: dict[str, Any]) -> Tier:
        return Tier.READ

    async def execute(self, params: dict[str, Any]) -> Outcome:
        text = params.get("text")
        if not text:
            return Outcome(ok=False, error="text is required")
        return Outcome(ok=True, status="200", payload={"text": text})
