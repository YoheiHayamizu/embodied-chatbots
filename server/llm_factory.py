"""LLM provider factory.

Selects a Pipecat LLM service implementation based on the LLM_PROVIDER
environment variable. Supported values:

    anthropic  -> Claude Haiku 4.5
    openai     -> GPT-4o-mini
    google     -> Gemini 2.0 Flash
"""

import os
from typing import Literal

from loguru import logger

Provider = Literal["anthropic", "openai", "google"]

SYSTEM_INSTRUCTION = (
    "You are a voice assistant embodied in a Unitree G1 humanoid robot. "
    "Your responses are spoken aloud through a text-to-speech system. "
    "Follow these rules strictly:\n"
    "- Reply in English only.\n"
    "- Do not use emojis, bullet points, markdown, or code blocks.\n"
    "- Keep replies to one or two short sentences.\n"
    "- Ask a clarifying question when the request is ambiguous.\n"
    "- Do not narrate internal reasoning."
)

DEFAULT_MODELS: dict[Provider, str] = {
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


def create_llm_service(provider: str = "anthropic"):
    """Build the configured LLM service.

    Returns a Pipecat LLM service instance ready to drop into a Pipeline.
    """
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, ""))  # type: ignore[arg-type]

    logger.info(f"LLM provider={provider} model={model}")

    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=model,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            api_key=_require_env("OPENAI_API_KEY"),
            settings=OpenAILLMService.Settings(
                model=model,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    if provider == "google":
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=_require_env("GOOGLE_API_KEY"),
            settings=GoogleLLMService.Settings(
                model=model,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected one of: anthropic, openai, google."
    )
