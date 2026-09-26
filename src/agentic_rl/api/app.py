from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agentic_rl import __version__
from agentic_rl.api.routes import router
from agentic_rl.capabilities.base import Capability
from agentic_rl.capabilities.delegate import DelegateCapability
from agentic_rl.capabilities.generate_report import GenerateReportCapability
from agentic_rl.capabilities.http_call import HttpCallCapability
from agentic_rl.capabilities.registry import CapabilityRegistry
from agentic_rl.capabilities.run_code import DockerCodeRunner, RunCodeCapability, SubprocessCodeRunner, docker_available
from agentic_rl.capabilities.schedule_task import ScheduleTaskCapability
from agentic_rl.capabilities.web_search import DdgsSearchPort, WebSearchCapability
from agentic_rl.core import observability
from agentic_rl.core.agent import Agent
from agentic_rl.core.config import Settings, get_settings
from agentic_rl.core.hub import KnowledgeHub
from agentic_rl.core.memory import Consolidator, MemoryStore
from agentic_rl.core.models import AgentProfile
from agentic_rl.core.router import Router
from agentic_rl.core.session import SessionStore
from agentic_rl.core.store import EpisodeStore
from agentic_rl.core.team import Team
from agentic_rl.llm.base import Planner
from agentic_rl.llm.distiller import Distiller, MockDistiller
from agentic_rl.llm.mock import MockPlanner, heuristic_default_fn
from agentic_rl.llm.summarizer import MockSummarizer, Summarizer
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


def _build_policy(settings: Settings, policy_id: str | None = None) -> Policy:
    return {
        "linucb": LinUCBPolicy,
        "epsilon": EpsilonGreedyPolicy,
        "greedy": GreedyPolicy,
    }.get(policy_id or settings.policy, LinUCBPolicy)()


def load_profiles(settings: Settings) -> list[AgentProfile]:
    """The team's agents, from settings.agents_file (a JSON list of AgentProfile
    objects — see agents.example.json), or the single implicit default agent."""
    if settings.agents_file is None:
        return [AgentProfile()]
    raw = json.loads(Path(settings.agents_file).read_text(encoding="utf-8"))
    profiles = [AgentProfile.model_validate(item) for item in raw]
    if not profiles:
        raise ValueError(f"{settings.agents_file}: needs at least one agent")
    ids = [p.id for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{settings.agents_file}: duplicate agent ids in {ids}")
    return profiles


def _build_distiller(settings: Settings) -> Distiller:
    if settings.planner == "mock":
        return MockDistiller()
    from agentic_rl.llm.distiller import LLMDistiller

    return LLMDistiller(provider=settings.planner, model=settings.llm_model)


def _build_summarizer(settings: Settings) -> Summarizer:
    if settings.planner == "mock":
        return MockSummarizer()
    from agentic_rl.llm.summarizer import LLMSummarizer

    return LLMSummarizer(provider=settings.planner, model=settings.llm_model)


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

    # Forward reference: the scheduler needs a runner that calls the team, but the
    # team's agents need a registry that needs the scheduler (for schedule_task) —
    # resolved with a mutable box filled in once `team` exists, just below.
    agent_box: dict[str, Team] = {}

    async def _scheduled_runner(instruction: str) -> None:
        with observability.trace("scheduler.run", input=instruction, source="scheduler"):
            await agent_box["team"].run(instruction, source="scheduler")

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

    # One planner/distiller/summarizer serve every agent (they're stateless); each
    # agent gets its own policy, persisted under its own id. See docs/MULTI-AGENT-PLAN.md.
    planner = _build_planner(settings)
    profiles = load_profiles(settings)
    is_team = len(profiles) > 1

    memory = MemoryStore(store.connection)
    # in a team, a correction's rule starts private to the agent that got it and is
    # promoted once a peer's feedback independently agrees (core/memory.py)
    consolidator = Consolidator(memory, _build_distiller(settings), default_scope="private" if is_team else "team")
    sessions = SessionStore(store.connection)
    summarizer = _build_summarizer(settings)
    hub = KnowledgeHub(
        store.connection,
        prior=settings.trust_prior,
        beta=settings.trust_beta,
        min_obs=settings.trust_min_obs,
        enabled=settings.share_knowledge and is_team,
    )

    agents: list[Agent] = []
    delegates: list[DelegateCapability] = []
    for profile in profiles:
        policy = _build_policy(settings, profile.policy)
        saved_state = store.load_policy_state(policy.id, agent_id=profile.id)
        if saved_state is not None:
            policy.load_state(saved_state)
        extras: list[Capability] = []
        if is_team and settings.delegation_enabled:
            delegate = DelegateCapability(profile.id, profiles, max_depth=settings.max_delegation_depth)
            delegates.append(delegate)
            extras.append(delegate)
        agents.append(
            Agent(
                planner,
                policy,
                registry,
                store,
                settings,
                memory,
                consolidator,
                sessions,
                summarizer,
                profile=profile,
                hub=hub if is_team else None,
                extra_capabilities=extras,
            )
        )
        hub.register(profile.id, policy, share=profile.share)

    agent_router = Router(profiles, prior_weight=settings.router_prior_weight) if is_team else None
    team = Team(agents, store, settings, hub, agent_router)
    for delegate in delegates:
        delegate.bind(team.delegate)
    agent_box["team"] = team

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
    app.state.team = team
    app.state.agent = team.default_agent  # back-compat: the single-agent entry point
    app.state.hub = hub
    app.state.registry = registry
    app.state.memory = memory
    app.state.sessions = sessions
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
