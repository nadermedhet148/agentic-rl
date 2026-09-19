from __future__ import annotations

import json

import httpx


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path

    if path == "/api/data" and request.method == "GET":
        # Requires an explicit Accept: application/json header — mirrors the
        # ScriptedUser's hidden preference in sim/user.py, so implicit reward
        # (exec ok/fail) and explicit reward (the user's grade) point the same way.
        if request.headers.get("accept") != "application/json":
            return httpx.Response(400, json={"error": "Accept header required"})
        return httpx.Response(200, json={"items": [1, 2, 3]})

    if path == "/orders" and request.method == "POST":
        body = json.loads(request.content or b"{}")
        return httpx.Response(201, json={"id": 42, "received": body})

    return httpx.Response(404, json={"error": f"no route for {request.method} {path}"})


def make_mock_transport() -> httpx.MockTransport:
    return httpx.MockTransport(_handler)


def make_client() -> httpx.AsyncClient:
    """An httpx.AsyncClient wired to the fake environment — no real network calls."""
    return httpx.AsyncClient(transport=make_mock_transport())
