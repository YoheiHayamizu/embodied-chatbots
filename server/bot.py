"""Embodied voice agent pipeline.

The robot itself is the audio I/O: the local microphone feeds STT, and the
local speaker plays the TTS output. A transcript broadcaster forwards RTVI-like
events to any attached WebSocket client so developers can watch the dialog on
a phone or laptop.

Echo mitigation is layered:
  1) OS-level AEC (PipeWire module-echo-cancel) — handled outside this file.
  2) `AudioGateProcessor` — drops microphone audio while the bot is speaking,
     so the TTS is never fed back into STT even if the AEC leaks a bit.
  3) Interruptions disabled — the bot finishes speaking before accepting input.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy

from llm_factory import create_llm_service

load_dotenv(override=True)

LLM_PROVIDER = "anthropic"  # or "openai" or "google"
PIPER_VOICE = "en_US-ryan-high"
PIPER_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "piper"


class AudioGateProcessor(FrameProcessor):
    """Drop microphone audio while the bot is speaking.

    This keeps the TTS output from leaking into STT even when room acoustics
    are imperfect. Placed immediately downstream of transport.input(), so no
    audio reaches the STT service between BotStartedSpeakingFrame and
    BotStoppedSpeakingFrame.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        if self._bot_speaking and isinstance(frame, InputAudioRawFrame):
            return  # swallow the frame

        await self.push_frame(frame, direction)


class TranscriptBroadcaster(FrameProcessor):
    """Forward transcription / bot-output events to an async callback.

    The callback receives dicts shaped like RTVI messages so the browser UI
    can reuse the same rendering code it already has.
    """

    def __init__(self, publish: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        super().__init__()
        self._publish = publish

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._emit(
                "user-transcription",
                {
                    "text": frame.text,
                    "final": True,
                    "timestamp": frame.timestamp,
                    "user_id": frame.user_id,
                },
            )
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._emit("bot-llm-started", None)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._emit("bot-llm-stopped", None)
        elif isinstance(frame, LLMTextFrame):
            await self._emit("bot-llm-text", {"text": frame.text})
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._emit("bot-started-speaking", None)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._emit("bot-stopped-speaking", None)

        await self.push_frame(frame, direction)

    async def _emit(self, msg_type: str, data: Any) -> None:
        payload = {"label": "rtvi-ai", "type": msg_type, "data": data}
        try:
            await self._publish(payload)
        except Exception as exc:
            logger.warning(f"transcript publish failed: {exc}")


async def run_bot(
    publish: Callable[[dict[str, Any]], Awaitable[None]],
    stop_event: asyncio.Event,
) -> None:
    """Run the embodied voice agent until `stop_event` is set."""
    logger.info("Starting embodied bot (LocalAudioTransport)")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = WhisperSTTService(
        device="auto",
        compute_type="int8",
        settings=WhisperSTTService.Settings(
            model="distil-medium.en",
            language=Language.EN,
        ),
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=PIPER_VOICE),
        download_dir=str(PIPER_MODEL_DIR),
        use_cuda=False,
    )

    llm = create_llm_service(provider=LLM_PROVIDER)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Disable interruptions so the bot never interrupts itself on echo.
            turn_start_strategy=VADUserTurnStartStrategy(enable_interruptions=False),
        ),
    )

    audio_gate = AudioGateProcessor()
    transcript = TranscriptBroadcaster(publish)

    pipeline = Pipeline(
        [
            transport.input(),
            audio_gate,  # drop mic audio while bot is speaking
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            transcript,  # broadcasts user + bot text to WebSocket listeners
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    context.add_message(
        {"role": "developer", "content": "Greet the user in one short sentence."}
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False)
    run_task = asyncio.create_task(runner.run(task))
    stop_task = asyncio.create_task(stop_event.wait())

    done, _ = await asyncio.wait(
        {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if stop_task in done and not run_task.done():
        logger.info("Stop event received; cancelling bot task")
        await task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass


def _debug_publisher(payload: dict[str, Any]) -> Awaitable[None]:
    async def _noop() -> None:
        print(json.dumps(payload))
    return _noop()


if __name__ == "__main__":
    # Standalone smoke test — prints transcript events to stdout.
    stop = asyncio.Event()
    try:
        asyncio.run(run_bot(_debug_publisher, stop))
    except KeyboardInterrupt:
        pass
