from __future__ import annotations

import json

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from agentic_rl.core import observability
from agentic_rl.core.models import Candidate, State
from agentic_rl.llm.llm_planner import LLMPlanner
from agentic_rl.llm.mock import MockPlanner, heuristic_default_fn
from agentic_rl.llm.prompts import render_corrections, render_rules

# --- MockPlanner -------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_planner_matches_rule_by_substring():
    fetch_candidates = [Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})]
    planner = MockPlanner(rules=[("fetch", fetch_candidates)], default=[])

    result = await planner.plan(State(request="please fetch https://x"), [], [])

    assert result[0].capability == "http_call"
    # returned candidates are copies, not the same objects as the fixture
    assert result[0] is not fetch_candidates[0]


@pytest.mark.asyncio
async def test_mock_planner_falls_back_to_default():
    default = [Candidate(capability="schedule_task", params={"instruction": "noop"})]
    planner = MockPlanner(rules=[("fetch", [])], default=default)

    result = await planner.plan(State(request="do something else entirely"), [], [])

    assert result[0].capability == "schedule_task"


@pytest.mark.asyncio
async def test_mock_planner_default_fn_receives_state():
    def default_fn(state: State) -> list[Candidate]:
        return [Candidate(capability="http_call", params={"method": "GET", "url": state.request})]

    planner = MockPlanner(default_fn=default_fn)
    result = await planner.plan(State(request="https://example.com"), [], [])

    assert result[0].params["url"] == "https://example.com"


# --- heuristic_default_fn -----------------------------------------------------


def test_heuristic_default_fn_infers_get_from_url():
    result = heuristic_default_fn(State(request="fetch https://httpbin.org/json please"))
    assert result[0].capability == "http_call"
    assert result[0].params["method"] == "GET"
    assert result[0].params["url"] == "https://httpbin.org/json"
    assert result[0].needs_confirmation is False


def test_heuristic_default_fn_infers_post_from_wording():
    result = heuristic_default_fn(State(request="create a new order at https://api.example.com/orders"))
    assert result[0].params["method"] == "POST"
    assert result[0].needs_confirmation is True


def test_heuristic_default_fn_infers_delete_from_wording():
    result = heuristic_default_fn(State(request="delete the record at https://api.example.com/items/5"))
    assert result[0].params["method"] == "DELETE"


def test_heuristic_default_fn_strips_trailing_punctuation_from_url():
    result = heuristic_default_fn(State(request="check https://example.com/x, then report back."))
    assert result[0].params["url"] == "https://example.com/x"


def test_heuristic_default_fn_infers_schedule_task():
    result = heuristic_default_fn(State(request="schedule a daily health check"))
    assert result[0].capability == "schedule_task"
    assert result[0].needs_confirmation is True


def test_heuristic_default_fn_falls_back_to_no_candidates():
    assert heuristic_default_fn(State(request="what's the weather like")) == []


# --- prompts -----------------------------------------------------------------


def test_render_corrections_empty():
    assert render_corrections([]) == ""


def test_render_corrections_renders_bullets():
    rendered = render_corrections(["always confirm deletes", "use UTC"])
    assert "always confirm deletes" in rendered
    assert "use UTC" in rendered
    assert rendered.startswith("\n\n")


def test_render_rules_empty():
    assert render_rules([]) == ""


def test_render_rules_renders_bullets():
    rendered = render_rules(["always send Accept: application/json to api.example.com"])
    assert "Accept: application/json" in rendered
    assert "Standing rules" in rendered
    assert rendered.startswith("\n\n")


# --- LLMPlanner (PydanticAI Agent + provider Model) ---------------------------
#
# These default to the claude provider/model, but exercise the provider-agnostic
# `plan()` path — see "provider dispatch" below for tests specific to selecting
# openai/google. LLMPlanner is injected with a PydanticAI `Model` test double
# instead of a fake raw provider client — each real provider Model calls its own
# SDK's request method internally (e.g. AnthropicModel calls
# `client.beta.messages.create(...)`), which a hand-rolled fake can't stand in
# for reliably. `TestModel` is used where only the returned structured output
# matters; `FunctionModel` where the test needs to inspect the outgoing prompt
# or simulate a specific provider error.


def _plan_response_json(candidates: list[dict]) -> str:
    return json.dumps({"candidates": candidates})


@pytest.mark.asyncio
async def test_claude_planner_returns_parsed_candidates():
    model = TestModel(custom_output_args={"candidates": [{"capability": "http_call", "params": {"method": "GET", "url": "https://x"}}]})
    planner = LLMPlanner(model="claude-opus-5", pydantic_model=model)

    result = await planner.plan(State(request="fetch https://x"), [], [])

    assert len(result) == 1
    assert result[0] == Candidate(capability="http_call", params={"method": "GET", "url": "https://x"})


@pytest.mark.asyncio
async def test_claude_planner_includes_rules_before_corrections_in_prompt():
    captured: dict[str, list[ModelMessage]] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["messages"] = messages
        return ModelResponse(parts=[TextPart(content=_plan_response_json([]))])

    planner = LLMPlanner(pydantic_model=FunctionModel(capture))
    await planner.plan(
        State(request="fetch https://x"),
        [],
        prior_corrections=["past correction about x"],
        rules=["always send Accept: application/json"],
    )

    user_content = captured["messages"][0].parts[0].content
    assert "Standing rules" in user_content
    assert "always send Accept: application/json" in user_content
    assert "past correction about x" in user_content
    # rules come before corrections in the rendered prompt
    assert user_content.index("Standing rules") < user_content.index("past correction about x")


@pytest.mark.asyncio
async def test_claude_planner_handles_empty_candidates():
    model = TestModel(custom_output_args={"candidates": []})
    planner = LLMPlanner(pydantic_model=model)

    result = await planner.plan(State(request="???"), [], [])

    assert len(result) == 1
    assert result[0].confidence == 0.0
    assert result[0].needs_confirmation is True


@pytest.mark.asyncio
async def test_claude_planner_wraps_rate_limit_error():
    async def raise_rate_limit(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="claude-opus-5", body="slow down")

    planner = LLMPlanner(pydantic_model=FunctionModel(raise_rate_limit))

    with pytest.raises(RuntimeError, match="rate limited"):
        await planner.plan(State(request="fetch https://x"), [], [])


@pytest.mark.asyncio
async def test_claude_planner_wraps_not_found_error():
    async def raise_not_found(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=404, model_name="claude-opus-5", body="no such model")

    planner = LLMPlanner(pydantic_model=FunctionModel(raise_not_found))

    with pytest.raises(RuntimeError, match="model not found"):
        await planner.plan(State(request="fetch https://x"), [], [])


@pytest.mark.asyncio
async def test_claude_planner_wraps_generic_api_error():
    async def raise_server_error(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name="claude-opus-5", body="internal error")

    planner = LLMPlanner(pydantic_model=FunctionModel(raise_server_error))

    with pytest.raises(RuntimeError, match="api error \\(500\\)"):
        await planner.plan(State(request="fetch https://x"), [], [])


# --- provider dispatch (model-agnostic: end-to-end through LLMPlanner, not just
# llm/providers.py in isolation) ------------------------------------------------


def test_llm_planner_defaults_to_claude():
    planner = LLMPlanner()
    assert planner.id == "claude"


def test_llm_planner_id_reflects_selected_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    planner = LLMPlanner(provider="openai", model="gpt-5.1")
    assert planner.id == "openai"


def test_llm_planner_dispatches_to_openai_model(monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    planner = LLMPlanner(provider="openai", model="gpt-5.1")
    assert isinstance(planner._agent.model, OpenAIChatModel)


def test_llm_planner_dispatches_to_google_model(monkeypatch):
    from pydantic_ai.models.google import GoogleModel

    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    planner = LLMPlanner(provider="google", model="gemini-3-pro")
    assert isinstance(planner._agent.model, GoogleModel)


def test_llm_planner_openai_without_key_raises_at_construction(monkeypatch):
    # openai's raw SDK resolves credentials eagerly (unlike anthropic's) — this
    # is a real, deliberate difference documented in llm/providers.py, not a bug.
    from pydantic_ai.exceptions import UserError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(UserError):
        LLMPlanner(provider="openai", model="gpt-5.1")


# --- observability wiring (real Langfuse SDK, tracing disabled — no network) --


@pytest.mark.asyncio
async def test_claude_planner_traces_successful_call(monkeypatch):
    pytest.importorskip("langfuse")  # requires the `observability` extra
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-planner")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-planner")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "False")
    from agentic_rl.core.config import Settings

    original_client = observability._client
    observability.setup(Settings(observability_enabled=True))
    try:
        model = TestModel(
            custom_output_args={"candidates": [{"capability": "http_call", "params": {"method": "GET", "url": "https://x"}}]}
        )
        planner = LLMPlanner(pydantic_model=model)

        result = await planner.plan(State(request="fetch https://x"), [], [])

        assert len(result) == 1
        assert result[0].capability == "http_call"  # tracing must not change the return value
    finally:
        observability._client = original_client


@pytest.mark.asyncio
async def test_claude_planner_traces_error_without_swallowing_it(monkeypatch):
    pytest.importorskip("langfuse")  # requires the `observability` extra
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-planner")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-planner")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "False")
    from agentic_rl.core.config import Settings

    original_client = observability._client
    observability.setup(Settings(observability_enabled=True))
    try:

        async def raise_rate_limit(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(status_code=429, model_name="claude-opus-5", body="slow down")

        planner = LLMPlanner(pydantic_model=FunctionModel(raise_rate_limit))

        with pytest.raises(RuntimeError, match="rate limited"):
            await planner.plan(State(request="fetch https://x"), [], [])
    finally:
        observability._client = original_client
