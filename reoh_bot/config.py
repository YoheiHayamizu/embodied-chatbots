"""Runtime configuration for the REOH Daily voice bot.

All knobs flow through environment variables (loaded from ``.env`` at process
start) and are surfaced as a single immutable :class:`Settings` value so the
rest of the package never has to call ``os.getenv`` directly. This keeps the
pipeline pure-functional with respect to its config and makes it trivial to
override settings in tests by constructing a fresh ``Settings``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent

DEFAULT_PROMPT_PATH = _PACKAGE_ROOT / "prompts" / "e2lg_system_prompt.md"
DEFAULT_PIPER_MODEL_DIR = _REPO_ROOT / "models" / "piper"
# Scenarios live under <repo>/dataset/reoh/scenarios/ (a sibling of reoh_bot/).
# Override with REOH_SCENARIO_DIR if the dataset lives elsewhere.
DEFAULT_SCENARIO_DIR = _REPO_ROOT / "dataset" / "reoh" / "scenarios"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class STTSettings:
    """Whisper STT configuration."""

    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "int8"


@dataclass(frozen=True)
class TTSSettings:
    """Piper TTS configuration."""

    voice: str = "en_US-ryan-high"
    model_dir: Path = DEFAULT_PIPER_MODEL_DIR


@dataclass(frozen=True)
class LLMSettings:
    """Anthropic Claude configuration."""

    api_key: str
    model: str = "claude-haiku-4-5"


@dataclass(frozen=True)
class DailySettings:
    """Daily transport / REST configuration."""

    api_key: str
    api_url: str = "https://api.daily.co/v1"
    bot_name: str = "REOH Agent"
    room_url: str | None = None
    room_token: str | None = None
    room_expiry_seconds: int = 60 * 60


@dataclass(frozen=True)
class ScenarioSettings:
    """Scenario selection."""

    scenario_dir: Path = DEFAULT_SCENARIO_DIR
    scenario_id: str | None = None


@dataclass(frozen=True)
class Settings:
    """Top-level settings aggregating every subsystem.

    Construct with :meth:`from_env` at the edge of the process. Pass the
    instance down explicitly rather than re-reading the environment.
    """

    stt: STTSettings
    tts: TTSSettings
    llm: LLMSettings
    daily: DailySettings
    scenario: ScenarioSettings
    prompt_path: Path = DEFAULT_PROMPT_PATH
    host: str = "localhost"
    port: int = 7861

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` from process environment variables.

        Required keys:
          - ``ANTHROPIC_API_KEY``
          - ``DAILY_API_KEY`` *(unless ``DAILY_ROOM_URL`` is set, in which
            case the bot can join an existing room with just a token)*
        """
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        daily_key = os.getenv("DAILY_API_KEY", "")
        daily_room_url = os.getenv("DAILY_ROOM_URL")

        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env or export it.")
        if not daily_key and not daily_room_url:
            raise RuntimeError(
                "Either DAILY_API_KEY (to create rooms) or DAILY_ROOM_URL (to join an existing room) must be set."
            )

        return cls(
            stt=STTSettings(
                model=_env("STT_MODEL", "large-v3-turbo"),
                device=_env("STT_DEVICE", "auto"),
                compute_type=_env("STT_COMPUTE_TYPE", "int8"),
            ),
            tts=TTSSettings(
                voice=_env("PIPER_VOICE", "en_US-ryan-high"),
                model_dir=_env_path("PIPER_MODEL_DIR", DEFAULT_PIPER_MODEL_DIR),
            ),
            llm=LLMSettings(
                api_key=anthropic_key,
                model=_env("LLM_MODEL", "claude-haiku-4-5"),
            ),
            daily=DailySettings(
                api_key=daily_key,
                api_url=_env("DAILY_API_URL", "https://api.daily.co/v1"),
                bot_name=_env("REOH_BOT_NAME", "REOH Agent"),
                room_url=daily_room_url,
                room_token=os.getenv("DAILY_ROOM_TOKEN"),
                room_expiry_seconds=int(_env("DAILY_ROOM_EXPIRY_SECONDS", "3600")),
            ),
            scenario=ScenarioSettings(
                scenario_dir=_env_path("REOH_SCENARIO_DIR", DEFAULT_SCENARIO_DIR),
                scenario_id=os.getenv("REOH_SCENARIO_ID", "0"),
            ),
            prompt_path=_env_path("REOH_PROMPT_PATH", DEFAULT_PROMPT_PATH),
            host=_env("HOST", "localhost"),
            port=int(_env("PORT", "7861")),
        )
