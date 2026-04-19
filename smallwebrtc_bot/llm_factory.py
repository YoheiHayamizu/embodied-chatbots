"""LLM provider factory.

Selects a pipecat LLM service implementation based on the LLM_PROVIDER
environment variable. The shared system prompt lives here so that swapping
providers does not drift the bot's voice and style.

Supported providers:
  - anthropic -> AnthropicLLMService (default model: claude-haiku-4-5)
  - openai    -> OpenAILLMService    (default model: gpt-4o-mini)
  - google    -> GoogleLLMService    (default model: gemini-2.0-flash)
"""

from __future__ import annotations

import os
from typing import Literal

from loguru import logger

Provider = Literal["anthropic", "openai", "google"]

SYSTEM_PROMPT = (
    "You are the voice of a friendly humanoid robot assistant. "
    "Your words are spoken aloud through a text-to-speech system. "
    "Follow these rules strictly:\n"
    "- Reply in English only.\n"
    "- Do not use emojis, bullet points, markdown, or code fences.\n"
    "- Keep each reply to one or two short sentences.\n"
    "- Ask a clarifying question when the request is ambiguous.\n"
    "- Do not narrate internal reasoning or describe your own actions."
)

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"{key} is not set. Add it to .env or export it before starting the server."
        )
    return value


def build_llm(provider: str | None = None):
    """Build the configured LLM service.

    Args:
        provider: Override for the provider. Falls back to the ``LLM_PROVIDER``
            environment variable, then to ``"anthropic"``.

    Returns:
        A pipecat LLM service instance ready to insert into a ``Pipeline``.
    """
    resolved = (provider or os.getenv("LLM_PROVIDER") or "anthropic").lower()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(resolved, "")

    logger.info(f"LLM provider={resolved} model={model or '<service default>'}")

    if resolved == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=model or DEFAULT_MODELS["anthropic"],
                system_instruction=SYSTEM_PROMPT,
            ),
        )

    if resolved == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            api_key=_require_env("OPENAI_API_KEY"),
            settings=OpenAILLMService.Settings(
                model=model or DEFAULT_MODELS["openai"],
                system_instruction=SYSTEM_PROMPT,
            ),
        )

    if resolved == "google":
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=_require_env("GOOGLE_API_KEY"),
            settings=GoogleLLMService.Settings(
                model=model or DEFAULT_MODELS["google"],
                system_instruction=SYSTEM_PROMPT,
            ),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={resolved!r}. Expected one of: anthropic, openai, google."
    )
