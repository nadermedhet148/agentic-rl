from __future__ import annotations

from typing import Any

import httpx

from agentic_rl.capabilities.base import Capability, Outcome, Tier

READ_METHODS = {"GET", "HEAD", "OPTIONS"}
ALLOWED_METHODS = READ_METHODS | {"POST", "PUT", "PATCH", "DELETE"}

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "enum": sorted(ALLOWED_METHODS)},
        "url": {"type": "string", "description": "Absolute http(s) URL"},
        "headers": {"type": "object", "additionalProperties": {"type": "string"}},
        "json_body": {"description": "JSON-serializable request body, if any"},
    },
    "required": ["method", "url"],
    "additionalProperties": False,
}


class HttpCallCapability(Capability):
    """Makes an outbound HTTP request.

    Tier is READ for GET/HEAD/OPTIONS, WRITE for anything that can mutate remote
    state (POST/PUT/PATCH/DELETE) — write-tier calls go through the safety gate.
    """

    name = "http_call"
    description = "Make an HTTP request to a URL and return the response."
    input_schema = INPUT_SCHEMA

    def __init__(self, client: httpx.AsyncClient, timeout_s: float = 15.0, max_body_bytes: int = 256_000):
        self._client = client
        self._timeout_s = timeout_s
        self._max_body_bytes = max_body_bytes

    def tier_for(self, params: dict[str, Any]) -> Tier:
        method = str(params.get("method", "GET")).upper()
        return Tier.READ if method in READ_METHODS else Tier.WRITE

    async def execute(self, params: dict[str, Any]) -> Outcome:
        method = str(params.get("method", "GET")).upper()
        if method not in ALLOWED_METHODS:
            return Outcome(ok=False, error=f"unsupported method: {method}")
        url = params.get("url")
        if not url or not str(url).lower().startswith(("http://", "https://")):
            return Outcome(ok=False, error=f"invalid url: {url!r}")

        try:
            response = await self._client.request(
                method,
                str(url),
                headers=params.get("headers"),
                json=params.get("json_body"),
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            return Outcome(ok=False, error=f"{type(exc).__name__}: {exc}")

        body: Any
        raw = response.content[: self._max_body_bytes]
        try:
            body = response.json()
        except ValueError:
            body = raw.decode(response.encoding or "utf-8", errors="replace")

        ok = 200 <= response.status_code < 300
        return Outcome(
            ok=ok,
            status=str(response.status_code),
            payload=body,
            error=None if ok else f"http status {response.status_code}",
        )
