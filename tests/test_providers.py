from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel

from agentic_rl.llm import providers

# --- build_model: provider dispatch --------------------------------------------


def test_build_model_claude_returns_anthropic_model():
    model = providers.build_model("claude", "claude-opus-5")
    assert isinstance(model, AnthropicModel)


def test_build_model_claude_needs_no_api_key_at_construction(monkeypatch):
    # anthropic.AsyncAnthropic() resolves credentials lazily (at request time),
    # unlike OpenAI/Google's raw clients — see llm/providers.py module docstring.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = providers.build_model("claude", "claude-opus-5")
    assert isinstance(model, AnthropicModel)


def test_build_model_openai_returns_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    model = providers.build_model("openai", "gpt-5.1")
    assert isinstance(model, OpenAIChatModel)


def test_build_model_openai_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(UserError, match="OPENAI_API_KEY"):
        providers.build_model("openai", "gpt-5.1")


def test_build_model_google_returns_google_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-key")
    model = providers.build_model("google", "gemini-3-pro")
    assert isinstance(model, GoogleModel)


def test_build_model_google_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(UserError, match="GOOGLE_API_KEY"):
        providers.build_model("google", "gemini-3-pro")


def test_build_model_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        providers.build_model("mock", "whatever")


# --- build_model: cross-provider model-name guard -------------------------------


def test_build_model_rejects_claude_model_name_for_openai_provider():
    with pytest.raises(ValueError, match="looks like it belongs to a different provider"):
        providers.build_model("openai", "claude-opus-5")


def test_build_model_rejects_openai_model_name_for_claude_provider():
    with pytest.raises(ValueError, match="looks like it belongs to a different provider"):
        providers.build_model("claude", "gpt-5.1")


def test_build_model_rejects_google_model_name_for_openai_provider():
    with pytest.raises(ValueError, match="looks like it belongs to a different provider"):
        providers.build_model("openai", "gemini-3-pro")


def test_build_model_allows_unrecognized_model_name_prefix(monkeypatch):
    # only a *known cross-provider* mismatch is rejected — an unrecognized
    # prefix (future model naming, a fine-tune alias, ...) is not assumed wrong.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    model = providers.build_model("openai", "o5-mini")
    assert isinstance(model, OpenAIChatModel)


# --- build_settings ---------------------------------------------------------


def test_build_settings_claude():
    settings = providers.build_settings("claude", max_tokens=4096, effort="medium")
    assert isinstance(settings, dict)  # ModelSettings subclasses are TypedDicts
    assert settings["max_tokens"] == 4096
    assert settings["thinking"] is True
    assert settings["anthropic_effort"] == "medium"
    assert settings["anthropic_cache_instructions"] is True


def test_build_settings_openai():
    settings = providers.build_settings("openai", max_tokens=2048, effort="low")
    assert settings["max_tokens"] == 2048
    assert settings["thinking"] == "low"
    assert "anthropic_effort" not in settings


def test_build_settings_google():
    settings = providers.build_settings("google", max_tokens=1024, effort="high")
    assert settings["max_tokens"] == 1024
    assert settings["thinking"] is True


def test_build_settings_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        providers.build_settings("mock", max_tokens=100, effort="low")
