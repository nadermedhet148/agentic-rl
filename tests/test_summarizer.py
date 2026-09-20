from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from agentic_rl.core import observability
from agentic_rl.core.config import Settings
from agentic_rl.llm.prompts import render_turns_for_summary
from agentic_rl.llm.summarizer import LLMSummarizer, MockSummarizer

# --- render_turns_for_summary -------------------------------------------------


def test_render_turns_for_summary_empty():
    assert render_turns_for_summary([]) == ""


def test_render_turns_for_summary_renders_pairs():
    rendered = render_turns_for_summary([("what is langfuse?", "an observability platform")])
    assert "User: what is langfuse?" in rendered
    assert "Agent: an observability platform" in rendered


# --- MockSummarizer ------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_summarizer_folds_turns_onto_prior_summary():
    summarizer = MockSummarizer()
    summary = await summarizer.summarize("", [("what is langfuse?", "an observability platform")])
    assert "what is langfuse?" in summary
    assert "an observability platform" in summary


@pytest.mark.asyncio
async def test_mock_summarizer_appends_to_a_nonempty_prior_summary():
    summarizer = MockSummarizer()
    summary = await summarizer.summarize("earlier: discussed langfuse", [("now what?", "make it a PDF")])
    assert summary.startswith("earlier: discussed langfuse")
    assert "now what?" in summary


@pytest.mark.asyncio
async def test_mock_summarizer_truncates_long_output():
    summarizer = MockSummarizer()
    summary = await summarizer.summarize("x" * 3000, [])
    assert len(summary) <= 2000


# --- LLMSummarizer (PydanticAI Agent + provider Model) -------------------------
#
# Same rationale as test_planner.py's LLMPlanner section: each real provider Model
# calls its own SDK's request method internally, so LLMSummarizer is injected with
# a PydanticAI `Model` test double instead of a fake provider client.


@pytest.mark.asyncio
async def test_claude_summarizer_returns_parsed_result():
    model = TestModel(custom_output_args={"summary": "the user asked about langfuse and got an overview"})
    summarizer = LLMSummarizer(model="claude-opus-5", pydantic_model=model)

    result = await summarizer.summarize("", [("what is langfuse?", "an observability platform")])

    assert result == "the user asked about langfuse and got an overview"


@pytest.mark.asyncio
async def test_claude_summarizer_includes_prior_summary_and_turns_in_prompt():
    captured: dict[str, list[ModelMessage]] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["messages"] = messages
        return ModelResponse(parts=[TextPart(content='{"summary": "x"}')])

    summarizer = LLMSummarizer(pydantic_model=FunctionModel(capture))

    await summarizer.summarize(
        "earlier the user asked about langfuse", [("now make that a PDF", "generated langfuse_report.pdf")]
    )

    user_content = captured["messages"][0].parts[0].content
    assert "earlier the user asked about langfuse" in user_content
    assert "now make that a PDF" in user_content
    assert "generated langfuse_report.pdf" in user_content


@pytest.mark.asyncio
async def test_claude_summarizer_wraps_rate_limit_error():
    async def raise_rate_limit(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="claude-opus-5", body="slow down")

    summarizer = LLMSummarizer(pydantic_model=FunctionModel(raise_rate_limit))

    with pytest.raises(RuntimeError, match="rate limited"):
        await summarizer.summarize("", [("x", "y")])


# --- provider dispatch (model-agnostic) -----------------------------------


def test_llm_summarizer_id_reflects_selected_provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    summarizer = LLMSummarizer(provider="google", model="gemini-3-pro")
    assert summarizer.id == "google"


def test_llm_summarizer_dispatches_to_google_model(monkeypatch):
    from pydantic_ai.models.google import GoogleModel

    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    summarizer = LLMSummarizer(provider="google", model="gemini-3-pro")
    assert isinstance(summarizer._agent.model, GoogleModel)


# --- observability wiring (real Langfuse SDK, tracing disabled — no network) --


@pytest.mark.asyncio
async def test_claude_summarizer_traces_successful_call(monkeypatch):
    pytest.importorskip("langfuse")  # requires the `observability` extra
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-summarizer")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-summarizer")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "False")

    original_client = observability._client
    observability.setup(Settings(observability_enabled=True))
    try:
        model = TestModel(custom_output_args={"summary": "a summary"})
        summarizer = LLMSummarizer(pydantic_model=model)

        result = await summarizer.summarize("", [("x", "y")])

        assert result == "a summary"  # tracing must not change the return value
    finally:
        observability._client = original_client
