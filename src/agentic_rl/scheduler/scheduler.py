from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# APScheduler's SQLAlchemyJobStore pickles job references by import path, so the job
# function must be a plain module-level callable (not a bound method or closure). A
# process-global dispatch target is the standard workaround; this app runs one
# AgentScheduler per process, so that's fine.
_dispatch_target: AgentScheduler | None = None


def _run_scheduled_instruction(instruction: str) -> None:
    if _dispatch_target is not None:
        _dispatch_target._dispatch(instruction)


class AgentScheduler:
    """Implements capabilities.schedule_task.SchedulerPort on top of APScheduler.

    Jobs are persisted to `db_path` (same SQLite file as the episode store, different
    tables) so scheduled tasks survive a restart. When a job fires it re-runs the
    stored natural-language instruction through the agent loop with source="scheduler"
    — see core/agent.py — so whatever the policy has learned by then applies.
    """

    def __init__(
        self,
        db_path: str | Path,
        agent_runner: Callable[[str], Awaitable[Any]],
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        global _dispatch_target
        if str(db_path) == ":memory:":
            jobstores = {"default": MemoryJobStore()}
        else:
            jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{Path(db_path).as_posix()}")}
        self._scheduler = BackgroundScheduler(jobstores=jobstores)
        self._agent_runner = agent_runner
        self._loop = loop
        _dispatch_target = self

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once the app's asyncio event loop is running (see api/app.py) so
        scheduler-thread jobs can dispatch back onto it."""
        self._loop = loop

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self, wait: bool = False) -> None:
        self._scheduler.shutdown(wait=wait)

    def _dispatch(self, instruction: str) -> None:
        if self._loop is None:
            asyncio.run(self._agent_runner(instruction))
        else:
            asyncio.run_coroutine_threadsafe(self._agent_runner(instruction), self._loop)

    # --- SchedulerPort ----------------------------------------------------------

    def add_job(
        self,
        instruction: str,
        *,
        cron: str | None = None,
        run_at: str | None = None,
        timezone: str | None = None,
    ) -> str:
        if bool(cron) == bool(run_at):
            raise ValueError("exactly one of cron or run_at must be set")
        trigger = (
            CronTrigger.from_crontab(cron, timezone=timezone)
            if cron
            else DateTrigger(run_date=run_at, timezone=timezone)
        )
        job = self._scheduler.add_job(_run_scheduled_instruction, trigger, args=[instruction])
        return job.id

    def remove_job(self, job_id: str) -> None:
        self._scheduler.remove_job(job_id)

    def list_jobs(self) -> list[dict]:
        jobs = self._scheduler.get_jobs()
        return [
            {
                "id": job.id,
                "instruction": job.args[0] if job.args else None,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in jobs
        ]
