from __future__ import annotations

from typing import Any, Protocol

from agentic_rl.capabilities.base import Capability, Outcome, Tier

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "Natural-language instruction to re-run through the agent when the job fires.",
        },
        "cron": {
            "type": "string",
            "description": "Cron expression (5-field, e.g. '0 9 * * *'). Mutually exclusive with run_at.",
        },
        "run_at": {
            "type": "string",
            "description": "ISO-8601 timestamp for a one-off run. Mutually exclusive with cron.",
        },
        "timezone": {"type": "string", "description": "IANA timezone, defaults to UTC."},
    },
    "required": ["instruction"],
    "additionalProperties": False,
}


class SchedulerPort(Protocol):
    """What ScheduleTaskCapability needs from the scheduler.

    Kept as a narrow protocol (rather than importing the scheduler module directly)
    so the capability can be unit-tested with a fake and so capabilities/ doesn't
    depend on scheduler/ or core/agent.py (which depends on capabilities/), avoiding
    an import cycle.
    """

    def add_job(
        self,
        instruction: str,
        *,
        cron: str | None = None,
        run_at: str | None = None,
        timezone: str | None = None,
    ) -> str:
        """Schedule a job, returning its id."""
        ...


class ScheduleTaskCapability(Capability):
    """Schedules a deferred re-run of the agent loop with a natural-language instruction.

    The instruction (not a frozen action) is what's stored, so that by the time the
    job fires the planner/policy can apply anything learned since it was scheduled.
    """

    name = "schedule_task"
    description = (
        "Schedule a natural-language instruction to be run again later, either once "
        "(run_at) or on a recurring cron schedule."
    )
    input_schema = INPUT_SCHEMA

    def __init__(self, scheduler: SchedulerPort):
        self._scheduler = scheduler

    def tier_for(self, params: dict[str, Any]) -> Tier:
        return Tier.WRITE

    async def execute(self, params: dict[str, Any]) -> Outcome:
        instruction = params.get("instruction")
        cron = params.get("cron")
        run_at = params.get("run_at")
        timezone = params.get("timezone")

        if not instruction:
            return Outcome(ok=False, error="instruction is required")
        if bool(cron) == bool(run_at):
            return Outcome(ok=False, error="exactly one of cron or run_at must be set")

        try:
            job_id = self._scheduler.add_job(
                instruction, cron=cron, run_at=run_at, timezone=timezone
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a normal failed outcome
            return Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")

        return Outcome(ok=True, status="scheduled", payload={"job_id": job_id})
