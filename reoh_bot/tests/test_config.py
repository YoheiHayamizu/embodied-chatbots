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
