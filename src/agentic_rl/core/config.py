from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(StrEnum):
    """Controls how much the policy may explore and when writes need confirmation.

    sim         - full exploration, no confirmation gate (simulator only)
    dev         - explore on read-tier actions only; confirm write-tier when planner asks
    prod_strict - greedy; every write-tier action requires confirmation
    """

    SIM = "sim"
    DEV = "dev"
    PROD_STRICT = "prod_strict"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_RL_", env_file=".env", extra="ignore")

    mode: Mode = Mode.DEV
    db_path: Path = Path("agentic_rl.db")
    llm_model: str = "claude-opus-5"
    # claude | openai | google | mock — see llm/providers.py. Selects both the
    # planner and the distiller (one flag drives both — see api/app.py). When
    # changing this away from "claude", also set llm_model to a model id from
    # that provider; see .env.example.
    planner: str = "claude"
    policy: str = "linucb"  # linucb | epsilon | greedy
    http_timeout_s: float = 15.0
    http_max_body_bytes: int = 256_000
    web_search_max_results: int = 5
    web_search_timeout_s: float = 10.0
    corrections_top_k: int = 5
    max_steps: int = 6  # cap on plan->act->observe iterations per episode (see core/agent.py)
    planner_history_max_chars: int = 4000  # per-step outcome payload truncation in the planner prompt
    run_code_timeout_s: float = 10.0
    run_code_max_output_bytes: int = 64_000
    run_code_docker_image: str = "python:3.12-slim"
    run_code_memory_mb: int = 256
    reports_dir: Path = Path("reports")
    session_summarize_every: int = 5  # fold conversation turns into the rolling summary this often

    # Multi-agent team (docs/MULTI-AGENT-PLAN.md). With no agents_file, the team is
    # one implicit "default" agent with every capability — the single-agent setup.
    agents_file: Path | None = None  # JSON list of AgentProfile objects; see agents.example.json
    share_knowledge: bool = True  # master switch for agents learning from each other
    trust_prior: float = 0.5  # trust in a peer before there's evidence either way
    trust_beta: float = 0.1  # EMA rate for learned trust (core/hub.py)
    trust_min_obs: int = 2  # observations before learned trust replaces the prior
    peer_min_trust: float = 0.2  # peers trusted less than this don't feed corrections/examples
    demonstrations_top_k: int = 3  # peers' approved examples shown to the planner
    router_prior_weight: float = 0.5  # weight of the capability-cue prior in routing (core/router.py)
    delegation_enabled: bool = True
    max_delegation_depth: int = 1  # a delegated child may not delegate again
    delegation_credit: float = 0.5  # share of a parent's feedback passed to its child episode

    # Langfuse tracing (see docs/OBSERVABILITY.md) — off by default. Everything
    # else Langfuse needs (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL,
    # ...) is read directly from the environment by the langfuse SDK itself, not
    # proxied through this Settings object.
    observability_enabled: bool = False


def get_settings() -> Settings:
    return Settings()
