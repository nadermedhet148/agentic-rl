from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agentic_rl import __version__
from agentic_rl.api.routes import router
from agentic_rl.capabilities.generate_report import GenerateReportCapability
from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.run_code import DockerCodeRunner, RunCodeCapability, SubprocessCodeRunner, docker_available
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.capabilities.web_search import DdgsSearchPort, WebSearchCapability
from agentic_rl.core import observability
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Settings, get_settings
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.llm.base import Planner
from agentic_rl.llm.distiller import Distiller, MockDistiller
from agentic_rl.llm.mock import MockPlanner, heuristic_default_fn
from agentic_rl.policy.base import Policy
from agentic_rl.policy.epsilon import EpsilonGreedyPolicy
from agentic_rl.policy.greedy import GreedyPolicy
from agentic_rl.policy.linucb import LinUCBPolicy
from agentic_rl.scheduler.scheduler import AgentScheduler

STATIC_DIR = Path(__file__).parent / "static"


def _build_planner(settings: Settings) -> Planner:
    if settings.planner == "mock":
        return MockPlanner(default_fn=heuristic_default_fn)
    # local import: only construct a provider client if a real LLM is actually needed
    from agentic_rl.llm.llm_planner import LLMPlanner

    return LLMPlanner(
        provider=settings.planner,
        model=settings.llm_model,
        history_max_chars=settings.planner_history_max_chars,
        max_steps=settings.max_steps,
    )


def _build_policy(settings: Settings) -> Policy:
    return {
        "linucb": LinUCBPolicy,
        "epsilon": EpsilonGreedyPolicy,
        "greedy": GreedyPolicy,
    }.get(settings.policy, LinUCBPolicy)()


def _build_distiller(settings: Settings) -> Distiller:
    if settings.planner == "mock":
        return MockDistiller()
    from agentic_rl.llm.distiller import LLMDistiller

    return LLMDistiller(provider=settings.planner, model=settings.llm_model)


def create_app(settings: Settings | None = None) -> FastAPI:
    # Loads .env into the real process environment (os.environ), not just this
    # module's Settings fields — pydantic-settings' own `env_file=".env"` (see
    # core/config.py) only populates Settings' own fields; ANTHROPIC_API_KEY and
    # the LANGFUSE_* vars are read directly from os.environ by their SDKs, so
    # they need this too. Never overrides a var already set in the environment
    # (e.g. by run.bat, a shell export, or a container's env) — see .env.example.
    load_dotenv()

    settings = settings or get_settings()
    observability.setup(settings)  # no-op unless AGENTIC_RL_OBSERVABILITY_ENABLED — see docs/OBSERVABILITY.md

    http_client = httpx.AsyncClient()
    store = EpisodeStore(settings.db_path)

    # Forward reference: the scheduler needs a runner that calls the agent, but the
    # agent needs a registry that needs the scheduler (for schedule_task) — resolved
    # with a mutable box filled in once `agent` exists, just below.
    agent_box: dict[str, Agent] = {}

    async def _scheduled_runner(instruction: str) -> None:
        with observability.trace("scheduler.run", input=instruction, source="scheduler"):
            await agent_box["agent"].run(instruction, source="scheduler")

    scheduler = AgentScheduler(settings.db_path, _scheduled_runner)

    registry = CapabilityRegistry()
    registry.register(
        HttpCallCapability(http_client, timeout_s=settings.http_timeout_s, max_body_bytes=settings.http_max_body_bytes)
    )
    registry.register(ScheduleTaskCapability(scheduler))
    registry.register(
        WebSearchCapability(
            DdgsSearchPort(),
            max_results=settings.web_search_max_results,
            timeout_s=settings.web_search_timeout_s,
        )
    )
    code_runner = (
        DockerCodeRunner(settings.run_code_docker_image, settings.run_code_memory_mb)
        if docker_available()
        else SubprocessCodeRunner()
    )
    registry.register(
        RunCodeCapability(
            code_runner,
            timeout_s=settings.run_code_timeout_s,
            max_output_bytes=settings.run_code_max_output_bytes,
        )
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    registry.register(GenerateReportCapability(settings.reports_dir))

    planner = _build_planner(settings)
    policy = _build_policy(settings)
    saved_state = store.load_policy_state(policy.id)
    if saved_state is not None:
        policy.load_state(saved_state)

    memory = MemoryStore(store.connection)
    consolidator = Consolidator(memory, _build_distiller(settings))

    agent = Agent(planner, policy, registry, store, settings, memory, consolidator)
    agent_box["agent"] = agent

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler.set_loop(asyncio.get_running_loop())
        scheduler.start()
        try:
            yield
        finally:
            scheduler.shutdown()
            await http_client.aclose()
            store.close()
            observability.flush()
            observability.shutdown()

    app = FastAPI(title="agentic-rl", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.scheduler = scheduler
    app.state.agent = agent
    app.state.registry = registry
    app.state.memory = memory
    app.state.background_tasks = set()  # keeps SSE worker tasks (api/routes.py) alive

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": settings.mode, "version": __version__}

    app.include_router(router)

    # Registered before the catch-all "/" mount below — Starlette matches mounts in
    # registration order and Mount("/") prefix-matches every path, so anything added
    # after it would never be reached.
    app.mount("/reports", StaticFiles(directory=str(settings.reports_dir)), name="reports")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
