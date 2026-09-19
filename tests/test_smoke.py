from fastapi.testclient import TestClient

from agentic_rl.api.app import create_app
from agentic_rl.core.config import Mode, Settings


def test_health():
    app = create_app(Settings(mode=Mode.SIM, db_path=":memory:", planner="mock"))
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["mode"] == "sim"
