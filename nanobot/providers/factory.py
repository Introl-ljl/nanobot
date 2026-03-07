"""Provider factory helpers."""

from __future__ import annotations

from nanobot.config.schema import Config


def create_provider(
    config: Config,
    model: str | None = None,
    *,
    set_global_api_base: bool = False,
    allow_missing_standard_credentials: bool = False,
):
    """Create an LLM provider instance for the given model."""
    from nanobot.providers.custom_provider import CustomProvider
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.registry import find_by_name

    selected_model = (model or config.agents.defaults.model).strip() or config.agents.defaults.model
    provider_name = config.get_provider_name(selected_model)
    provider_config = config.get_provider(selected_model)
    is_bedrock_model = selected_model.lower().startswith("bedrock/")

    if provider_name == "openai_codex" or selected_model.startswith(("openai-codex/", "openai_codex/")):
        return OpenAICodexProvider(default_model=selected_model)

    if provider_name == "custom":
        return CustomProvider(
            api_key=provider_config.api_key if provider_config and provider_config.api_key else "no-key",
            api_base=config.get_api_base(selected_model) or "http://localhost:8000/v1",
            default_model=selected_model,
        )

    spec = find_by_name(provider_name) if provider_name else None
    if spec and spec.is_oauth:
        return LiteLLMProvider(
            api_key=None,
            api_base=config.get_api_base(selected_model),
            default_model=selected_model,
            provider_name=provider_name,
            set_global_api_base=set_global_api_base,
        )

    if not is_bedrock_model and not allow_missing_standard_credentials and not (provider_config and provider_config.api_key):
        raise ValueError("No API key configured for the selected model.")

    return LiteLLMProvider(
        api_key=provider_config.api_key if provider_config else None,
        api_base=config.get_api_base(selected_model),
        default_model=selected_model,
        extra_headers=provider_config.extra_headers if provider_config else None,
        provider_name=provider_name,
        set_global_api_base=set_global_api_base,
    )
