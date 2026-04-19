"""Pipecat voice-agent pipeline served over Daily.

Mirrors ``smallwebrtc_bot/bot.py`` but swaps the ``SmallWebRTCTransport`` for
``DailyTransport`` and routes generation through a scenario-bound
:class:`reoh_bot.e2lg_agent.E2LGAgent` instead of a generic LLM factory.

Lifecycle:

* :func:`run_bot` is invoked once per Daily room. It builds the pipeline,
  greets the first participant when they join, and tears the pipeline down on
  participant departure or session expiry.
* Audio: Whisper STT -> E2LG (Claude) -> Piper TTS, with Silero VAD on the
  user-aggregator so partial transcripts only commit on end-of-turn.
"""

from __future__ import annotations

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from reoh_bot.config import Settings
from reoh_bot.e2lg_agent import E2LGAgent, E2LGModelSettings, load_prompt_template
from reoh_bot.scenarios import load_scenario


def _build_stt(settings: Settings) -> WhisperSTTService:
    logger.info(
        "STT model={} device={} compute_type={}",
        settings.stt.model,
        settings.stt.device,
        settings.stt.compute_type,
    )
    return WhisperSTTService(
        device=settings.stt.device,
        compute_type=settings.stt.compute_type,
        settings=WhisperSTTService.Settings(
            model=settings.stt.model,
            language=Language.EN,
        ),
    )


def _build_tts(settings: Settings) -> PiperTTSService:
    settings.tts.model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TTS voice={} model_dir={}", settings.tts.voice, settings.tts.model_dir)
    return PiperTTSService(
        settings=PiperTTSService.Settings(voice=settings.tts.voice),
        download_dir=settings.tts.model_dir,
    )


def _build_agent(settings: Settings) -> E2LGAgent:
    scenario = load_scenario(settings.scenario.scenario_dir, settings.scenario.scenario_id)
    prompt_template = load_prompt_template(settings.prompt_path)
    return E2LGAgent.from_scenario(
        scenario=scenario,
        settings=E2LGModelSettings(
            api_key=settings.llm.api_key,
            model=settings.llm.model,
        ),
        prompt_template=prompt_template,
    )


async def run_bot(*, settings: Settings, room_url: str, token: str | None) -> None:
    """Run the REOH voice agent for a single Daily room.

    Args:
        settings: Fully-resolved process settings.
        room_url: URL of the Daily room to join.
        token: Optional meeting token granting bot privileges.
    """
    logger.info("Starting REOH bot in room {}", room_url)

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name=settings.daily.bot_name,
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            microphone_out_enabled=True,
            transcription_enabled=False,
        ),
    )

    stt = _build_stt(settings)
    tts = _build_tts(settings)
    agent = _build_agent(settings)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            agent.llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # Daily Prebuilt sends its own app-messages (chat, presence) over the
        # transport's data channel. Pipecat's auto-injected RTVIProcessor would
        # try to parse all of them as RTVI protocol messages, then echo
        # validation errors back into the room — a feedback loop. We're not
        # using an RTVI client here, so disable it entirely.
        enable_rtvi=False,
    )

    @transport.event_handler("on_first_participant_joined")
    async def _on_first_participant_joined(_transport: DailyTransport, participant: dict) -> None:
        logger.info(
            "First participant joined scenario={} participant={}",
            agent.scenario.scenario_id,
            participant.get("id"),
        )
        context.add_message(agent.opening_directive())
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def _on_participant_left(_transport: DailyTransport, participant: dict, reason: str) -> None:
        logger.info("Participant left ({}); ending pipeline", reason)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception:
        logger.exception("REOH bot pipeline crashed")
    finally:
        logger.info("REOH bot finished for room {}", room_url)
