from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from agentic_rl.core import observability
from agentic_rl.core.config import Settings
from agentic_rl.core.models import Action, Candidate, Episode, Memory, Outcome, State
from agentic_rl.llm.distiller import DistillResult, LLMDistiller, MockDistiller
from agentic_rl.llm.prompts import render_existing_rules


def make_episode() -> Episode:
    candidate = Candidate(capability="http_call", params={"method": "GET", "url": "https://api.example.com/orders"})
    action = Action(candidate=candidate, index=0, explored=False, arm_id="http_call:abc:False")
    return Episode(
        state=State(request="fetch orders from https://api.example.com/orders"),
        candidates=[candidate],
        action=action,
        outcome=Outcome(ok=True, status="200"),
        implicit_reward=0.2,
        planner_id="mock",
        policy_id="linucb",
    )


# --- render_existing_rules ---------------------------------------------------


def test_render_existing_rules_empty():
    assert "none yet" in render_existing_rules([])


def test_render_existing_rules_renders_id_and_text():
    rendered = render_existing_rules([("abc123", "always use json")])
    assert "[abc123]" in rendered
    assert "always use json" in rendered


# --- MockDistiller -------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_distiller_creates_rule_verbatim():
    distiller = MockDistiller()
    result = await distiller.distill("always confirm deletes", make_episode(), [])
    assert result.rule_text == "always confirm deletes"
    assert result.matches_existing_id is None
    assert result.supersedes_id is None


@pytest.mark.asyncio
async def test_mock_distiller_matches_normalized_text():
    existing = [Memory(text="  always   confirm deletes  ")]
    distiller = MockDistiller()
    result = await distiller.distill("Always Confirm Deletes", make_episode(), existing)
    assert result.matches_existing_id == existing[0].id


@pytest.mark.asyncio
async def test_mock_distiller_never_supersedes():
    existing = [Memory(text="a completely different rule")]
    distiller = MockDistiller()
    result = await distiller.distill("always confirm deletes", make_episode(), existing)
    assert result.supersedes_id is None
    assert result.matches_existing_id is None


# --- LLMDistiller (PydanticAI Agent + provider Model) --------------------------
#
# Same rationale as test_planner.py's LLMPlanner section: each real provider
# Model calls its own SDK's request method internally, so LLMDistiller is
# injected with a PydanticAI `Model` test double instead of a fake provider
# client. Note there's no "handles None parsed output" test anymore —
# PydanticAI's `result.output` is a validated instance of `output_type`, never
# None.


@pytest.mark.asyncio
async def test_claude_distiller_returns_parsed_result():
    model = TestModel(
        custom_output_args={"rule_text": "always send Accept: application/json", "capability": "http_call"}
    )
    distiller = LLMDistiller(model="claude-opus-5", pydantic_model=model)

    result = await distiller.distill("always send Accept: application/json", make_episode(), [])

    assert result == DistillResult(rule_text="always send Accept: application/json", capability="http_call")


@pytest.mark.asyncio
async def test_claude_distiller_includes_existing_rules_in_prompt():
    captured: dict[str, list[ModelMessage]] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["messages"] = messages
        return ModelResponse(parts=[TextPart(content='{"rule_text": "x"}')])

    distiller = LLMDistiller(pydantic_model=FunctionModel(capture))

    existing = [Memory(id="rule-1", text="always use UTC")]
    await distiller.distill("some correction", make_episode(), existing)

    user_content = captured["messages"][0].parts[0].content
    assert "[rule-1]" in user_content
    assert "always use UTC" in user_content
    assert "some correction" in user_content


@pytest.mark.asyncio
async def test_claude_distiller_wraps_rate_limit_error():
    async def raise_rate_limit(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="claude-opus-5", body="slow down")

    distiller = LLMDistiller(pydantic_model=FunctionModel(raise_rate_limit))

    with pytest.raises(RuntimeError, match="rate limited"):
        await distiller.distill("some correction", make_episode(), [])


# --- provider dispatch (model-agnostic) -----------------------------------


def test_llm_distiller_id_reflects_selected_provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    distiller = LLMDistiller(provider="google", model="gemini-3-pro")
    assert distiller.id == "google"


def test_llm_distiller_dispatches_to_google_model(monkeypatch):
    from pydantic_ai.models.google import GoogleModel

    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    distiller = LLMDistiller(provider="google", model="gemini-3-pro")
    assert isinstance(distiller._agent.model, GoogleModel)


# --- observability wiring (real Langfuse SDK, tracing disabled — no network) --


@pytest.mark.asyncio
async def test_claude_distiller_traces_successful_call(monkeypatch):
    pytest.importorskip("langfuse")  # requires the `observability` extra
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-distiller")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-distiller")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "False")

    original_client = observability._client
    observability.setup(Settings(observability_enabled=True))
    try:
        model = TestModel(custom_output_args={"rule_text": "always send Accept: application/json"})
        distiller = LLMDistiller(pydantic_model=model)

        result = await distiller.distill("always send Accept: application/json", make_episode(), [])

        assert result.rule_text == "always send Accept: application/json"  # tracing must not change the return value
    finally:
        observability._client = original_client
