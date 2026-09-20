from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import respx

from agentic_rl.capabilities.answer import AnswerCapability
from agentic_rl.capabilities.base import Tier
from agentic_rl.capabilities.generate_report import GenerateReportCapability, _safe_filename
from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.run_code import RunCodeCapability, SubprocessCodeRunner, docker_available
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.capabilities.web_search import WebSearchCapability


def make_http_capability() -> HttpCallCapability:
    client = httpx.AsyncClient()
    return HttpCallCapability(client)


class FakeSearch:
    def __init__(self, results: list[dict] | None = None, error: Exception | None = None) -> None:
        self._results = results if results is not None else []
        self._error = error
        self.calls: list[tuple[str, int]] = []

    def text(self, query: str, max_results: int) -> list[dict]:
        self.calls.append((query, max_results))
        if self._error is not None:
            raise self._error
        return self._results


class FakeRunner:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.calls: list[str] = []

    async def run(self, script_path, workdir) -> tuple[int, bytes, bytes]:
        self.calls.append(script_path.read_text(encoding="utf-8"))
        return self._returncode, self._stdout, self._stderr


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


# --- web_search --------------------------------------------------------------


def test_web_search_tier_is_read():
    cap = WebSearchCapability(FakeSearch())
    assert cap.tier_for({"query": "x"}) is Tier.READ


@pytest.mark.asyncio
async def test_web_search_success():
    search = FakeSearch(
        results=[{"title": "Example", "href": "https://example.com", "body": "an example"}]
    )
    cap = WebSearchCapability(search, max_results=3)
    outcome = await cap.execute({"query": "example"})
    assert outcome.ok is True
    assert outcome.status == "200"
    assert outcome.payload["results"] == [
        {"title": "Example", "url": "https://example.com", "snippet": "an example"}
    ]
    assert search.calls == [("example", 3)]


@pytest.mark.asyncio
async def test_web_search_requires_query():
    cap = WebSearchCapability(FakeSearch())
    outcome = await cap.execute({})
    assert outcome.ok is False
    assert "query is required" in outcome.error


@pytest.mark.asyncio
async def test_web_search_surfaces_backend_error():
    cap = WebSearchCapability(FakeSearch(error=RuntimeError("boom")))
    outcome = await cap.execute({"query": "example"})
    assert outcome.ok is False
    assert "boom" in outcome.error


# --- answer --------------------------------------------------------------


def test_answer_tier_is_read():
    cap = AnswerCapability()
    assert cap.tier_for({"text": "hi"}) is Tier.READ


@pytest.mark.asyncio
async def test_answer_success():
    cap = AnswerCapability()
    outcome = await cap.execute({"text": "the answer is 42"})
    assert outcome.ok is True
    assert outcome.payload == {"text": "the answer is 42"}


@pytest.mark.asyncio
async def test_answer_requires_text():
    cap = AnswerCapability()
    outcome = await cap.execute({})
    assert outcome.ok is False
    assert "text is required" in outcome.error


# --- run_code ----------------------------------------------------------------


def test_run_code_tier_is_write():
    cap = RunCodeCapability(FakeRunner())
    assert cap.tier_for({"code": "print(1)"}) is Tier.WRITE


@pytest.mark.asyncio
async def test_run_code_success_returns_stdout():
    runner = FakeRunner(returncode=0, stdout=b"5050\n")
    cap = RunCodeCapability(runner)
    outcome = await cap.execute({"code": "print(sum(range(101)))"})
    assert outcome.ok is True
    assert outcome.status == "0"
    assert outcome.payload["stdout"] == "5050\n"
    assert outcome.payload["sandbox"] == "subprocess"
    assert "sum(range(101))" in runner.calls[0]


@pytest.mark.asyncio
async def test_run_code_nonzero_exit_surfaces_stderr_as_error():
    runner = FakeRunner(returncode=1, stderr=b"Traceback: boom\n")
    cap = RunCodeCapability(runner)
    outcome = await cap.execute({"code": "raise RuntimeError('boom')"})
    assert outcome.ok is False
    assert outcome.status == "1"
    assert "boom" in outcome.error


@pytest.mark.asyncio
async def test_run_code_requires_code():
    cap = RunCodeCapability(FakeRunner())
    outcome = await cap.execute({})
    assert outcome.ok is False
    assert "code is required" in outcome.error


@pytest.mark.asyncio
async def test_run_code_truncates_output():
    runner = FakeRunner(returncode=0, stdout=b"x" * 100)
    cap = RunCodeCapability(runner, max_output_bytes=10)
    outcome = await cap.execute({"code": "print('x' * 100)"})
    assert outcome.payload["stdout"] == "x" * 10


def test_docker_available_false_when_docker_not_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert docker_available() is False


def test_docker_available_false_when_daemon_unreachable(monkeypatch):
    import subprocess

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no daemon")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert docker_available() is False


def test_docker_available_true_when_info_succeeds(monkeypatch):
    import subprocess

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert docker_available() is True


@pytest.mark.asyncio
async def test_run_code_real_subprocess_runner():
    # No mocking — a real local subprocess, same spirit as test_store.py's real
    # sqlite-file tests: fast, no network, no external service.
    cap = RunCodeCapability(SubprocessCodeRunner())
    outcome = await cap.execute({"code": "print(1 + 1)"})
    assert outcome.ok is True
    assert "2" in outcome.payload["stdout"]
    assert outcome.payload["sandbox"] == "subprocess"


# --- generate_report -----------------------------------------------------


def test_generate_report_tier_is_write(tmp_path):
    cap = GenerateReportCapability(tmp_path)
    assert cap.tier_for({"title": "t", "content": "c"}) is Tier.WRITE


@pytest.mark.asyncio
async def test_generate_report_success_writes_pdf(tmp_path):
    cap = GenerateReportCapability(tmp_path)
    outcome = await cap.execute(
        {"title": "Weekly Update", "content": "# Summary\nAll good.\n\n- item one\n- item two"}
    )
    assert outcome.ok is True
    assert outcome.payload["filename"] == "Weekly-Update.pdf"
    assert outcome.payload["url"] == "/reports/Weekly-Update.pdf"
    saved = tmp_path / "Weekly-Update.pdf"
    assert saved.exists()
    assert outcome.payload["size_bytes"] == saved.stat().st_size


@pytest.mark.asyncio
async def test_generate_report_requires_title_and_content(tmp_path):
    cap = GenerateReportCapability(tmp_path)
    outcome = await cap.execute({"title": "", "content": "x"})
    assert outcome.ok is False
    assert "required" in outcome.error


def test_safe_filename_strips_path_traversal():
    assert _safe_filename("../../evil.pdf", "fallback title") == "evil.pdf"


def test_safe_filename_falls_back_to_title():
    assert _safe_filename(None, "Q3 Report!!") == "Q3-Report.pdf"


@pytest.mark.asyncio
async def test_generate_report_path_traversal_stays_inside_reports_dir(tmp_path):
    cap = GenerateReportCapability(tmp_path)
    outcome = await cap.execute({"title": "t", "content": "c", "filename": "../../evil.pdf"})
    assert outcome.ok is True
    assert outcome.payload["filename"] == "evil.pdf"
    assert (tmp_path / "evil.pdf").exists()
    assert not (tmp_path.parent / "evil.pdf").exists()


# --- registry --------------------------------------------------------------


def test_registry_register_and_lookup(tmp_path):
    registry = CapabilityRegistry()
    http_cap = make_http_capability()
    sched_cap = ScheduleTaskCapability(FakeScheduler())
    search_cap = WebSearchCapability(FakeSearch())
    answer_cap = AnswerCapability()
    run_code_cap = RunCodeCapability(FakeRunner())
    report_cap = GenerateReportCapability(tmp_path)
    registry.register(http_cap)
    registry.register(sched_cap)
    registry.register(search_cap)
    registry.register(answer_cap)
    registry.register(run_code_cap)
    registry.register(report_cap)

    expected = {"http_call", "schedule_task", "web_search", "answer", "run_code", "generate_report"}
    assert set(registry.names()) == expected
    assert registry.get("http_call") is http_cap

    with pytest.raises(KeyError):
        registry.get("nope")

    schemas = registry.tool_schemas()
    assert {s["name"] for s in schemas} == expected


def test_registry_rejects_duplicate_names():
    registry = CapabilityRegistry()
    registry.register(make_http_capability())
    with pytest.raises(ValueError):
        registry.register(make_http_capability())
