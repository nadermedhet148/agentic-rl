from __future__ import annotations

from agentic_rl.core.models import Candidate

# Hidden preferences the simulated user grades against — the agent never sees these
# directly, only the feedback they produce. Mirrors sim/env.py's fake endpoints.
API_DOMAIN_PATH = "/api/data"
ORDERS_PATH = "/orders"
PREFERRED_TIMEZONE = "Europe/Berlin"


class ScriptedUser:
    """Deterministically grades a chosen candidate against a fixed set of preferences,
    producing the same (score, correction) shape as a real Feedback."""

    def grade(self, candidate: Candidate) -> tuple[int, str | None]:
        if candidate.capability == "http_call":
            return self._grade_http_call(candidate)
        if candidate.capability == "schedule_task":
            return self._grade_schedule_task(candidate)
        return 0, None

    def _grade_http_call(self, candidate: Candidate) -> tuple[int, str | None]:
        method = str(candidate.params.get("method", "GET")).upper()
        url = str(candidate.params.get("url", ""))
        headers = {k.lower(): v for k, v in (candidate.params.get("headers") or {}).items()}

        if method == "GET" and API_DOMAIN_PATH in url:
            if headers.get("accept") != "application/json":
                return -1, "always include header Accept: application/json when calling /api/data"
            return 1, None

        if method == "POST" and ORDERS_PATH in url:
            if not candidate.needs_confirmation:
                return -1, "always require confirmation before POSTing to /orders"
            return 1, None

        return 0, None

    def _grade_schedule_task(self, candidate: Candidate) -> tuple[int, str | None]:
        if candidate.params.get("timezone") != PREFERRED_TIMEZONE:
            return -1, f"always schedule tasks in the {PREFERRED_TIMEZONE} timezone"
        return 1, None
