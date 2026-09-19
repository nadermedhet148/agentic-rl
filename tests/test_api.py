from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

from agentic_rl.api.app import create_app
from agentic_rl.core.config import Mode, Settings


def make_client(mode: Mode = Mode.DEV) -> TestClient:
    app = create_app(Settings(mode=mode, db_path=":memory:", planner="mock", policy="greedy"))
    return TestClient(app)


@respx.mock
def test_chat_executes_read_tier_and_returns_episode():
    respx.get("https://example.com").mock(return_value=httpx.Response(200, json={"ok": True}))
    with make_client() as client:
        r = client.post("/chat", json={"message": "http_call GET https://example.com"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("executed", "pending_confirmation")
    assert "id" in body


def test_chat_with_unrecognized_request_needs_confirmation():
    # MockPlanner() with no rules/default returns [] -> agent falls back to a
    # no-candidate placeholder that always needs confirmation.
    with make_client() as client:
        r = client.post("/chat", json={"message": "anything"})
    assert r.status_code == 200
    assert r.json()["status"] == "pending_confirmation"


def test_confirm_unknown_episode_returns_404():
    with make_client() as client:
        r = client.post("/confirm/does-not-exist")
    assert r.status_code == 404


def test_feedback_unknown_episode_returns_404():
    with make_client() as client:
        r = client.post("/feedback", json={"episode_id": "nope", "score": 1})
    assert r.status_code == 404


def test_chat_then_feedback_roundtrip():
    with make_client() as client:
        pending = client.post("/chat", json={"message": "anything"}).json()
        r = client.post("/feedback", json={"episode_id": pending["id"], "score": 1})
    assert r.status_code == 200
    assert r.json()["final_reward"] == 1.0


def test_episodes_and_metrics_endpoints():
    with make_client() as client:
        client.post("/chat", json={"message": "anything"})
        client.post("/chat", json={"message": "something else"})

        episodes = client.get("/episodes").json()
        assert len(episodes) == 2

        metrics = client.get("/metrics").json()
        assert metrics["n"] == 2


def test_tasks_endpoints_list_and_cancel():
    with make_client() as client:
        assert client.get("/tasks").json() == []
        r = client.delete("/tasks/does-not-exist")
    assert r.status_code == 404


def test_memories_add_list_delete():
    with make_client() as client:
        assert client.get("/memories").json() == []

        r = client.post("/memories", json={"text": "always use UTC", "capability": "schedule_task"})
        assert r.status_code == 200
        memory = r.json()
        assert memory["text"] == "always use UTC"
        assert memory["capability"] == "schedule_task"
        assert memory["support_count"] == 1

        listed = client.get("/memories").json()
        assert len(listed) == 1
        assert listed[0]["id"] == memory["id"]

        r = client.delete(f"/memories/{memory['id']}")
        assert r.status_code == 200
        assert client.get("/memories").json() == []


def test_delete_unknown_memory_returns_404():
    with make_client() as client:
        r = client.delete("/memories/does-not-exist")
    assert r.status_code == 404


def test_feedback_with_correction_creates_a_memory():
    with make_client() as client:
        pending = client.post("/chat", json={"message": "anything"}).json()
        client.post(
            "/feedback",
            json={"episode_id": pending["id"], "score": -1, "correction": "always confirm first"},
        )
        memories = client.get("/memories").json()
    assert len(memories) == 1
    assert memories[0]["text"] == "always confirm first"


# --- model-agnostic provider selection (AGENTIC_RL_PLANNER=openai|google) -----


def test_app_boots_with_openai_planner(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    app = create_app(Settings(db_path=":memory:", planner="openai", llm_model="gpt-5.1", policy="greedy"))
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200


def test_app_boots_with_google_planner(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    app = create_app(Settings(db_path=":memory:", planner="google", llm_model="gemini-3-pro", policy="greedy"))
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
