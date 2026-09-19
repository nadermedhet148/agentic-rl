from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def disable_observability_by_default_in_tests():
    """Ensure tests run with observability disabled by default, isolating tests
    from any local .env file where AGENTIC_RL_OBSERVABILITY_ENABLED may be true.
    Tests specifically verifying observability (like test_observability.py)
    use monkeypatch to enable it explicitly."""
    os.environ["AGENTIC_RL_OBSERVABILITY_ENABLED"] = "false"
