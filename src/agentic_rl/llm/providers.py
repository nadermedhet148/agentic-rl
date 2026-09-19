"""Builds a PydanticAI `Model` + `ModelSettings` for a given provider flag.

This is the one place that knows about `claude`/`openai`/`google` as concrete
things — `LLMPlanner`/`LLMDistiller` (llm/llm_planner.py, llm/distiller.py) are
otherwise provider-agnostic: they just hold a PydanticAI `Agent` and call
`.run()`. See docs/ARCHITECTURE.md "LLM calls" for the full design rationale.

Two things are deliberately *not* uniform across providers, verified against
each provider's installed SDK/provider class rather than assumed:

1. **Credential resolution at construction time.** `anthropic.AsyncAnthropic()`
   resolves its full credential chain (API key, `ant auth login` profiles, ...)
   lazily — it does not require a key just to construct. `openai.AsyncOpenAI()`
   and `google.genai.Client()` both raise immediately if no key is available.
   So only the Claude path constructs its own raw client and hands it to
   PydanticAI (preserving that lazy, multi-source resolution); OpenAI and
   Google are left to PydanticAI's own provider inference (`provider="openai"`
   / `"google"`), which already produces a clear, actionable error
   (`UserError: Set the OPENAI_API_KEY environment variable ...`) — there's no
   fuller chain to preserve underneath it. Concretely, this means constructing
   an `LLMPlanner`/`LLMDistiller` for openai/google **can raise at app startup**
   if the corresponding key isn't set; for claude it can't.

2. **Effort/reasoning control.** PydanticAI has one unified field for this,
   `ModelSettings.thinking` (`True`/`False`/`"minimal"`/`"low"`/`"medium"`/
   `"high"`/`"xhigh"`), but it doesn't carry full granularity for every
   provider: for Claude, any truthy value just switches on adaptive thinking
   (the specific level is dropped in that branch — read the `_translate_thinking`
   method in the installed `pydantic_ai.models.anthropic` if this ever needs
   re-verifying against a new SDK version) — so Claude additionally gets the
   provider-specific `anthropic_effort` field, which does carry the level.
   OpenAI's unified `thinking` *does* map onto its own reasoning-effort scale
   one-for-one (`OPENAI_REASONING_EFFORT_MAP`, same literal names), so no
   provider-specific override is needed there. Google gets `thinking=True`
   only — it has no comparable effort-level scale exposed here.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

SUPPORTED_PROVIDERS = ("claude", "openai", "google")

# A quick, cheap guard against the most likely misconfiguration — switching
# `planner` without also updating `llm_model` — rather than a confusing 404
# from the provider's API after a real request goes out.
_MODEL_ID_PREFIX_BY_PROVIDER = {
    "claude": "claude-",
    "openai": "gpt-",
    "google": "gemini-",
}


def _check_model_name(provider: str, model_name: str) -> None:
    expected_prefix = _MODEL_ID_PREFIX_BY_PROVIDER.get(provider)
    if expected_prefix is None:
        return
    other_prefixes = [p for prov, p in _MODEL_ID_PREFIX_BY_PROVIDER.items() if prov != provider]
    if model_name.startswith(expected_prefix):
        return
    if any(model_name.startswith(p) for p in other_prefixes):
        raise ValueError(
            f"llm_model={model_name!r} looks like it belongs to a different provider than "
            f"planner={provider!r} — set AGENTIC_RL_LLM_MODEL to a {provider} model id "
            f"(see .env.example)."
        )
    # An unrecognized prefix isn't necessarily wrong (model catalogs change) — only the
    # cross-provider mismatch above is treated as an error.


def build_model(provider: str, model_name: str) -> Model:
    """Construct the PydanticAI `Model` for `provider`. Raises `ValueError` for an
    unknown provider, or immediately (from PydanticAI itself) for openai/google if
    the corresponding API key isn't set — see module docstring point 1."""
    _check_model_name(provider, model_name)

    if provider == "claude":
        import anthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            model_name, provider=AnthropicProvider(anthropic_client=anthropic.AsyncAnthropic())
        )

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(model_name)  # provider="openai" (default) reads OPENAI_API_KEY

    if provider == "google":
        from pydantic_ai.models.google import GoogleModel

        return GoogleModel(model_name)  # provider="google" (default) reads GOOGLE_API_KEY

    raise ValueError(f"unknown LLM provider: {provider!r} (expected one of {SUPPORTED_PROVIDERS})")


def build_settings(provider: str, *, max_tokens: int, effort: str) -> ModelSettings:
    """`effort` is `"low"`/`"medium"`/`"high"` — this project's own scale, chosen
    to be valid on both Claude's and OpenAI's effort literals. See module
    docstring point 2 for why each provider is (or isn't) given a provider-
    specific override on top of the shared `thinking` field."""
    if provider == "claude":
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        return AnthropicModelSettings(
            max_tokens=max_tokens,
            thinking=True,
            anthropic_effort=effort,
            anthropic_cache_instructions=True,  # cache_control on the last system block
        )

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModelSettings

        return OpenAIChatModelSettings(max_tokens=max_tokens, thinking=effort)

    if provider == "google":
        from pydantic_ai.models.google import GoogleModelSettings

        return GoogleModelSettings(max_tokens=max_tokens, thinking=True)

    raise ValueError(f"unknown LLM provider: {provider!r} (expected one of {SUPPORTED_PROVIDERS})")
