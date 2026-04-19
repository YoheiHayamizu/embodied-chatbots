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
    """Whisper STT configuration.

    ``no_speech_prob`` is the threshold above which Whisper drops a segment
    as "probably silence". pipecat's default is 0.4, which is fine for the
    large models on GPU but too aggressive for ``tiny.en`` on CPU — the
    model frequently flags real speech as silence and ``run_stt`` emits no
    frame at all, leaving the user-aggregator stuck waiting. Bumping to 0.6
    lets borderline segments through. Tune higher (e.g. 0.8) if the bot
    still misses utterances on Jetson.
    """

    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "int8"
    no_speech_prob: float = 0.6


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
class TurnSettings:
    """User-turn detection thresholds.

    Two parameters control how long the bot waits between sentences:

    * ``vad_stop_secs`` — how long of continuous silence the VAD itself
      needs before declaring "user stopped speaking". Pipecat's default
      is 0.2s, which cuts visitors off mid-utterance during natural
      breath pauses. Bump to 0.8–1.0s for conversational speech.
    * ``speech_timeout`` — extra wait after VAD-stop, before the
      aggregator commits the turn. Acts as a safety net for the case
      where the visitor inhales and then keeps talking.

    Total natural pause budget = ``vad_stop_secs + speech_timeout``.

    ``stop_timeout`` is the safety fallback that fires if no strategy
    matches. It MUST exceed the worst-case STT processing latency,
    otherwise slow STT (e.g. on Jetson) finishes after the turn is
    already considered closed and the transcript is silently discarded.
    """

    speech_timeout: float = 0.6
    stop_timeout: float = 8.0
    vad_stop_secs: float = 0.8


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
    turn: TurnSettings
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
                no_speech_prob=float(_env("STT_NO_SPEECH_PROB", "0.6")),
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
            turn=TurnSettings(
                speech_timeout=float(_env("USER_SPEECH_TIMEOUT", "0.6")),
                stop_timeout=float(_env("USER_TURN_STOP_TIMEOUT", "8.0")),
                vad_stop_secs=float(_env("VAD_STOP_SECS", "0.8")),
            ),
            prompt_path=_env_path("REOH_PROMPT_PATH", DEFAULT_PROMPT_PATH),
            host=_env("HOST", "localhost"),
            port=int(_env("PORT", "7861")),
        )
