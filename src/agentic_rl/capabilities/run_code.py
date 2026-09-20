from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from agentic_rl.capabilities.base import Capability, Outcome, Tier

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python source to run. Printed output (stdout) is returned."},
    },
    "required": ["code"],
    "additionalProperties": False,
}


def docker_available() -> bool:
    """True only if the docker CLI is on PATH *and* its daemon actually answers.

    `shutil.which("docker")` alone isn't enough — Docker Desktop can be
    installed (the CLI on PATH) but not running, in which case every `docker
    run` fails with a connection error instead of falling back cleanly.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class CodeRunner(Protocol):
    """What RunCodeCapability needs to actually run a script.

    Kept as a narrow protocol (rather than hardcoding subprocess/docker calls
    into the capability) so tests can inject a fake instead of spawning real
    processes — same pattern as SearchPort (capabilities/web_search.py) and
    SchedulerPort (capabilities/schedule_task.py).
    """

    async def run(self, script_path: Path, workdir: Path) -> tuple[int, bytes, bytes]:
        """Run script_path with workdir as its cwd, returning (returncode, stdout, stderr)."""
        ...


class SubprocessCodeRunner:
    """Runs the script as a plain subprocess of the app's own Python.

    Weaker isolation than DockerCodeRunner: the script shares this process's
    OS-level permissions (filesystem, network) and installed packages —
    `-I` (isolated mode) only drops the user site-packages dir and
    PYTHONPATH/PYTHONHOME influence, it is not a security sandbox. Used as
    the fallback when Docker isn't available.
    """

    async def run(self, script_path: Path, workdir: Path) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(script_path),
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout, stderr


class DockerCodeRunner:
    """Runs the script inside a throwaway, network-isolated container.

    `--rm` discards the container after exit; `--network none` blocks
    outbound network access; `--memory`/`--cpus` bound resource usage. This
    is a real (partial) mitigation, not a strong guarantee — the container
    still runs with whatever `image` provides.
    """

    def __init__(self, image: str = "python:3.12-slim", memory_mb: int = 256, cpus: float = 1.0):
        self._image = image
        self._memory_mb = memory_mb
        self._cpus = cpus

    async def run(self, script_path: Path, workdir: Path) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            f"--memory={self._memory_mb}m",
            f"--cpus={self._cpus}",
            "-v",
            f"{workdir}:/workspace",
            "-w",
            "/workspace",
            self._image,
            "python",
            f"/workspace/{script_path.name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout, stderr


class RunCodeCapability(Capability):
    """Runs a short Python snippet to compute or transform data and returns
    its stdout. Always write-tier: real code execution, always confirmed
    (see llm/prompts.py's guidance to the planner) before it runs.
    """

    name = "run_code"
    description = (
        "Run a short Python snippet for computation or data transforms you can't do "
        "reliably yourself (exact math, sorting, reshaping JSON/CSV, etc.) and return "
        "its printed output. Runs in an isolated sandbox with no persistence between "
        "calls — don't use it to fetch URLs (use http_call/web_search for that)."
    )
    input_schema = INPUT_SCHEMA

    def __init__(self, runner: CodeRunner, timeout_s: float = 10.0, max_output_bytes: int = 64_000):
        self._runner = runner
        self._timeout_s = timeout_s
        self._max_output_bytes = max_output_bytes

    def tier_for(self, params: dict[str, Any]) -> Tier:
        return Tier.WRITE

    async def execute(self, params: dict[str, Any]) -> Outcome:
        code = params.get("code")
        if not code:
            return Outcome(ok=False, error="code is required")

        try:
            with TemporaryDirectory(prefix="agentic_rl_run_code_") as tmp:
                workdir = Path(tmp)
                script_path = workdir / "script.py"
                script_path.write_text(code, encoding="utf-8")
                returncode, stdout, stderr = await asyncio.wait_for(
                    self._runner.run(script_path, workdir), timeout=self._timeout_s
                )
        except TimeoutError:
            return Outcome(ok=False, error=f"run_code timed out after {self._timeout_s}s")
        except Exception as exc:  # noqa: BLE001 - surfaced as a normal failed outcome
            return Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")

        stdout_text = stdout[: self._max_output_bytes].decode("utf-8", errors="replace")
        stderr_text = stderr[: self._max_output_bytes].decode("utf-8", errors="replace")
        ok = returncode == 0
        sandbox = "docker" if isinstance(self._runner, DockerCodeRunner) else "subprocess"
        return Outcome(
            ok=ok,
            status=str(returncode),
            payload={"stdout": stdout_text, "stderr": stderr_text, "sandbox": sandbox},
            error=None if ok else (stderr_text.strip()[:500] or f"exit code {returncode}"),
        )
