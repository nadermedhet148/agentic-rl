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
    corrections_top_k: int = 5

    # Langfuse tracing (see docs/OBSERVABILITY.md) — off by default. Everything
    # else Langfuse needs (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL,
    # ...) is read directly from the environment by the langfuse SDK itself, not
    # proxied through this Settings object.
    observability_enabled: bool = False


def get_settings() -> Settings:
    return Settings()
