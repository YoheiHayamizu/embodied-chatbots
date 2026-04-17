"""Smoke tests for the LLM provider factory.

These tests avoid hitting any provider network call by monkey-patching
``os.environ`` and checking that the returned service is the expected class.
Each test expects the pipecat provider package to be importable via the
installed ``pipecat-ai[anthropic,openai,google]`` extras.
"""

from __future__ import annotations

import pytest

from server.llm_factory import build_llm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_anthropic_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from pipecat.services.anthropic.llm import AnthropicLLMService

    svc = build_llm("anthropic")
    assert isinstance(svc, AnthropicLLMService)


def test_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from pipecat.services.openai.llm import OpenAILLMService

    svc = build_llm("openai")
    assert isinstance(svc, OpenAILLMService)


def test_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from pipecat.services.google.llm import GoogleLLMService

    svc = build_llm("google")
    assert isinstance(svc, GoogleLLMService)


def test_provider_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from pipecat.services.openai.llm import OpenAILLMService

    svc = build_llm()
    assert isinstance(svc, OpenAILLMService)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError):
        build_llm("anthropic")


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        build_llm("acme")
