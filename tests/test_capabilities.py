from __future__ import annotations

import httpx
import pytest
import respx

from agentic_rl.capabilities.base import Tier
from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability


def make_http_capability() -> HttpCallCapability:
    client = httpx.AsyncClient()
    return HttpCallCapability(client)


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self._next_id = 0

    def add_job(self, instruction, *, cron=None, run_at=None, timezone=None) -> str:
        self._next_id += 1
        job_id = f"job-{self._next_id}"
        self.jobs[job_id] = {
            "instruction": instruction,
            "cron": cron,
            "run_at": run_at,
            "timezone": timezone,
        }
        return job_id


# --- http_call ---------------------------------------------------------------


def test_http_call_tier_read_vs_write():
    cap = make_http_capability()
    assert cap.tier_for({"method": "GET", "url": "https://x"}) is Tier.READ
    assert cap.tier_for({"method": "HEAD", "url": "https://x"}) is Tier.READ
    assert cap.tier_for({"method": "POST", "url": "https://x"}) is Tier.WRITE
    assert cap.tier_for({"method": "DELETE", "url": "https://x"}) is Tier.WRITE


@pytest.mark.asyncio
@respx.mock
async def test_http_call_get_json_success():
    respx.get("https://api.example.com/thing").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    cap = make_http_capability()
    outcome = await cap.execute({"method": "GET", "url": "https://api.example.com/thing"})
    assert outcome.ok is True
    assert outcome.status == "200"
    assert outcome.payload == {"ok": True}


@pytest.mark.asyncio
@respx.mock
async def test_http_call_post_error_status():
    respx.post("https://api.example.com/orders").mock(return_value=httpx.Response(500, text="boom"))
    cap = make_http_capability()
    outcome = await cap.execute(
        {"method": "POST", "url": "https://api.example.com/orders", "json_body": {"a": 1}}
    )
    assert outcome.ok is False
    assert outcome.status == "500"
    assert "500" in outcome.error


@pytest.mark.asyncio
async def test_http_call_rejects_invalid_url():
    cap = make_http_capability()
    outcome = await cap.execute({"method": "GET", "url": "not-a-url"})
    assert outcome.ok is False
    assert "invalid url" in outcome.error


@pytest.mark.asyncio
async def test_http_call_rejects_bad_method():
    cap = make_http_capability()
    outcome = await cap.execute({"method": "TRACE", "url": "https://x"})
    assert outcome.ok is False
    assert "unsupported method" in outcome.error


# --- schedule_task -------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_task_cron():
    scheduler = FakeScheduler()
    cap = ScheduleTaskCapability(scheduler)
    outcome = await cap.execute({"instruction": "fetch the daily report", "cron": "0 9 * * *"})
    assert outcome.ok is True
    job_id = outcome.payload["job_id"]
    assert scheduler.jobs[job_id]["instruction"] == "fetch the daily report"
    assert scheduler.jobs[job_id]["cron"] == "0 9 * * *"


@pytest.mark.asyncio
async def test_schedule_task_requires_exactly_one_of_cron_or_run_at():
    cap = ScheduleTaskCapability(FakeScheduler())

    both = await cap.execute(
        {"instruction": "x", "cron": "0 9 * * *", "run_at": "2026-01-01T00:00:00Z"}
    )
    assert both.ok is False

    neither = await cap.execute({"instruction": "x"})
    assert neither.ok is False


def test_schedule_task_tier_is_write():
    cap = ScheduleTaskCapability(FakeScheduler())
    assert cap.tier_for({"instruction": "x", "cron": "0 9 * * *"}) is Tier.WRITE


# --- registry --------------------------------------------------------------


def test_registry_register_and_lookup():
    registry = CapabilityRegistry()
    http_cap = make_http_capability()
    sched_cap = ScheduleTaskCapability(FakeScheduler())
    registry.register(http_cap)
    registry.register(sched_cap)

    assert set(registry.names()) == {"http_call", "schedule_task"}
    assert registry.get("http_call") is http_cap

    with pytest.raises(KeyError):
        registry.get("nope")

    schemas = registry.tool_schemas()
    assert {s["name"] for s in schemas} == {"http_call", "schedule_task"}


def test_registry_rejects_duplicate_names():
    registry = CapabilityRegistry()
    registry.register(make_http_capability())
    with pytest.raises(ValueError):
        registry.register(make_http_capability())
