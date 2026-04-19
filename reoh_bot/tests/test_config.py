"""Unit tests for env-driven settings."""

from __future__ import annotations

import pytest

from reoh_bot.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ANTHROPIC_API_KEY",
        "DAILY_API_KEY",
        "DAILY_ROOM_URL",
        "DAILY_ROOM_TOKEN",
        "DAILY_API_URL",
        "DAILY_ROOM_EXPIRY_SECONDS",
        "REOH_BOT_NAME",
        "LLM_MODEL",
        "STT_MODEL",
        "STT_DEVICE",
        "STT_COMPUTE_TYPE",
        "PIPER_VOICE",
        "PIPER_MODEL_DIR",
        "REOH_SCENARIO_DIR",
        "REOH_SCENARIO_ID",
        "REOH_PROMPT_PATH",
        "HOST",
        "PORT",
        "REOH_AGENT_KIND",
        "REOH_SSLG_PROMPT_PATH",
        "REOH_PERSONA_ENABLED",
        "PERSONA_EXTRACTOR_MODEL",
        "PERSONA_EXTRACTOR_MAX_TOKENS",
        "PERSONA_EXTRACTOR_TIMEOUT_S",
        "PERSONA_MIN_UTTERANCE_TOKENS",
        "PERSONA_EXTRACTOR_PROMPT_PATH",
        "STRATEGY_WEIGHTS_JSON",
        "PERSONA_SELECTOR_SEED",
    ):
        monkeypatch.delenv(key, raising=False)


def test_from_env_requires_anthropic_key() -> None:
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        Settings.from_env()


def test_from_env_requires_daily_key_or_room(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    with pytest.raises(RuntimeError, match="DAILY_API_KEY"):
        Settings.from_env()


def test_from_env_accepts_room_url_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_ROOM_URL", "https://example.daily.co/room")
    settings = Settings.from_env()
    assert settings.daily.room_url == "https://example.daily.co/room"
    assert settings.daily.api_key == ""


def test_from_env_applies_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "daily-test")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("STT_DEVICE", "cuda")
    monkeypatch.setenv("PIPER_VOICE", "en_US-amy-medium")
    monkeypatch.setenv("PORT", "9000")

    settings = Settings.from_env()
    assert settings.llm.model == "claude-sonnet-4-6"
    assert settings.stt.device == "cuda"
    assert settings.tts.voice == "en_US-amy-medium"
    assert settings.port == 9000
    # Defaults still apply where unset.
    assert settings.daily.bot_name == "REOH Agent"
    assert settings.daily.room_expiry_seconds == 3600


def test_settings_are_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    settings = Settings.from_env()
    with pytest.raises(Exception):
        settings.llm.model = "other"  # type: ignore[misc]


def test_agent_kind_defaults_to_e2lg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    settings = Settings.from_env()
    assert settings.agent_kind == "e2lg"
    assert settings.persona.enabled is True
    assert settings.persona.extractor_model == "claude-haiku-4-5"


def test_agent_kind_sslg_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    monkeypatch.setenv("REOH_AGENT_KIND", "SSLG")  # case-insensitive
    settings = Settings.from_env()
    assert settings.agent_kind == "sslg"


def test_agent_kind_unknown_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    monkeypatch.setenv("REOH_AGENT_KIND", "banana")
    with pytest.raises(RuntimeError, match="REOH_AGENT_KIND"):
        Settings.from_env()


def test_persona_overrides_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    monkeypatch.setenv("REOH_PERSONA_ENABLED", "false")
    monkeypatch.setenv("PERSONA_EXTRACTOR_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("PERSONA_EXTRACTOR_TIMEOUT_S", "2.5")
    monkeypatch.setenv("PERSONA_MIN_UTTERANCE_TOKENS", "5")
    monkeypatch.setenv("PERSONA_SELECTOR_SEED", "42")
    settings = Settings.from_env()
    assert settings.persona.enabled is False
    assert settings.persona.extractor_model == "claude-sonnet-4-6"
    assert settings.persona.extractor_timeout_s == 2.5
    assert settings.persona.extractor_min_utterance_tokens == 5
    assert settings.persona.selector_seed == 42


def test_strategy_weights_json_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    monkeypatch.setenv(
        "STRATEGY_WEIGHTS_JSON",
        '{"logical_appeal": 0.5, "emotional_appeal": 0.2}',
    )
    settings = Settings.from_env()
    assert settings.persona.strategy_weights == {
        "logical_appeal": 0.5,
        "emotional_appeal": 0.2,
    }


def test_strategy_weights_json_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("DAILY_API_KEY", "test")
    monkeypatch.setenv("STRATEGY_WEIGHTS_JSON", "not json")
    with pytest.raises(RuntimeError, match="STRATEGY_WEIGHTS_JSON"):
        Settings.from_env()
