from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

from agentic_rl.api.app import create_app
from agentic_rl.core.config import Mode, Settings
from agentic_rl.core.store import EpisodeStore


@respx.mock
def test_policy_state_survives_app_restart(tmp_path):
    db_path = tmp_path / "agentic_rl.db"
    respx.get("https://example.com/thing").mock(return_value=httpx.Response(200, json={"ok": True}))

    settings = Settings(mode=Mode.DEV, db_path=db_path, planner="mock", policy="linucb")
    app1 = create_app(settings)
    with TestClient(app1) as client:
        r = client.post("/chat", json={"message": "anything"})
        episode = r.json()
        client.post("/feedback", json={"episode_id": episode["id"], "score": 1})

        r2 = client.post("/chat", json={"message": "http_call GET https://example.com/thing"})
        episode2 = r2.json()
        if episode2["status"] == "pending_confirmation":
            client.post(f"/confirm/{episode2['id']}")

    # the app's own EpisodeStore is closed by lifespan shutdown; verify via a fresh
    # connection that a policy_state row actually landed on disk.
    verify_store = EpisodeStore(db_path)
    saved_state = verify_store.load_policy_state("linucb")
    verify_store.close()
    assert saved_state is not None
    assert saved_state["arms"]  # at least one arm was learned about

    # rebuild the app against the same db file — a fresh LinUCBPolicy should come up
    # pre-loaded with that state rather than starting from a blank prior.
    app2 = create_app(Settings(mode=Mode.DEV, db_path=db_path, planner="mock", policy="linucb"))
    with TestClient(app2) as client:
        loaded_policy = app2.state.agent._policy
        assert loaded_policy.state_dict() == saved_state

        # episodes recorded before the restart are still there too
        episodes = client.get("/episodes").json()
        assert len(episodes) == 2
