from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from agentic_rl.scheduler.scheduler import AgentScheduler


def make_scheduler(agent_runner=None):
    calls = []

    async def default_runner(instruction: str):
        calls.append(instruction)

    scheduler = AgentScheduler(":memory:", agent_runner or default_runner)
    scheduler.start()
    return scheduler, calls


def test_add_job_cron_appears_in_list_jobs():
    scheduler, _ = make_scheduler()
    try:
        job_id = scheduler.add_job("daily report", cron="0 9 * * *")
        jobs = scheduler.list_jobs()
        assert any(j["id"] == job_id and j["instruction"] == "daily report" for j in jobs)
    finally:
        scheduler.shutdown()


def test_add_job_requires_exactly_one_of_cron_or_run_at():
    scheduler, _ = make_scheduler()
    try:
        with pytest.raises(ValueError):
            scheduler.add_job("x")
        with pytest.raises(ValueError):
            scheduler.add_job("x", cron="0 9 * * *", run_at="2030-01-01T00:00:00")
    finally:
        scheduler.shutdown()


def test_remove_job():
    scheduler, _ = make_scheduler()
    try:
        run_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        job_id = scheduler.add_job("future thing", run_at=run_at)
        assert any(j["id"] == job_id for j in scheduler.list_jobs())

        scheduler.remove_job(job_id)
        assert not any(j["id"] == job_id for j in scheduler.list_jobs())
    finally:
        scheduler.shutdown()


def test_scheduled_job_fires_and_dispatches_instruction():
    scheduler, calls = make_scheduler()
    try:
        run_at = datetime.now(UTC) + timedelta(seconds=0.2)
        scheduler.add_job("ping", run_at=run_at.isoformat())

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not calls:
            time.sleep(0.05)

        assert calls == ["ping"]
    finally:
        scheduler.shutdown()
