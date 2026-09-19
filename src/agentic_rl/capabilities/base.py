from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class Tier(StrEnum):
    """Safety tier of a capability invocation.

    READ  - no side effects; safe to explore/execute without confirmation.
    WRITE - side-effecting or hard to reverse; gated by the safety gate.
    """

    READ = "read"
    WRITE = "write"


class Outcome(BaseModel):
    ok: bool
    status: str | None = None
    payload: Any = None
    error: str | None = None


class Capability(ABC):
    """A predefined action the agent can take.

    Subclasses declare a JSON schema for their params (used to build the planner's
    tool definitions) and implement `execute`. `tier_for` lets a capability's safety
    tier depend on its params (e.g. http_call: GET is read, POST/PUT/DELETE is write).
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def tier_for(self, params: dict[str, Any]) -> Tier: ...

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> Outcome: ...
