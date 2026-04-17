"""Pipecat voice agent pipeline served over a SmallWebRTC peer connection.

The pipeline mirrors the shape of the local-audio reference in ``main.py``
(Whisper STT -> LLM -> Piper TTS) but swaps ``LocalAudioTransport`` for
``SmallWebRTCTransport`` so a browser client can stream audio in and out
via WebRTC. An ``RTVIProcessor`` plus ``RTVIObserver`` are inserted so the
browser receives transcription, LLM, and speaking events over the data
channel and can render a live transcript.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import (
    RTVIObserver,
    RTVIObserverParams,
    RTVIProcessor,
)
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from server.llm_factory import build_llm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PIPER_MODEL_DIR = _REPO_ROOT / "models" / "piper"


def _piper_model_dir() -> Path:
    configured = os.getenv("PIPER_MODEL_DIR")
    return Path(configured).expanduser() if configured else _DEFAULT_PIPER_MODEL_DIR


def _build_stt() -> WhisperSTTService:
    model = os.getenv("STT_MODEL", "large-v3-turbo")
    device = os.getenv("STT_DEVICE", "auto")
    compute_type = os.getenv("STT_COMPUTE_TYPE", "int8")
    logger.info(f"STT model={model} device={device} compute_type={compute_type}")
    return WhisperSTTService(
        device=device,
        compute_type=compute_type,
        settings=WhisperSTTService.Settings(
            model=model,
            language=Language.EN,
        ),
    )


def _build_tts() -> PiperTTSService:
    voice = os.getenv("PIPER_VOICE", "en_US-ryan-high")
    model_dir = _piper_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"TTS voice={voice} model_dir={model_dir}")
    return PiperTTSService(
        settings=PiperTTSService.Settings(voice=voice),
        download_dir=str(model_dir),
    )


async def run_bot(webrtc_connection: SmallWebRTCConnection) -> None:
    """Run the voice agent pipeline for one WebRTC peer connection.

    The coroutine returns when the pipeline task completes, which happens
    either because the peer disconnects or because the pipeline is cancelled
    from the app shutdown hook.
    """
    logger.info(f"Starting bot for pc_id={webrtc_connection.pc_id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = _build_stt()
    tts = _build_tts()
    llm = build_llm()

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    rtvi = RTVIProcessor(transport=transport)

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[RTVIObserver(rtvi, params=RTVIObserverParams())],
    )

    @rtvi.event_handler("on_client_ready")
    async def _on_client_ready(_rtvi: RTVIProcessor) -> None:
        logger.info("RTVI client ready; sending greeting")
        await _rtvi.set_bot_ready()
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the user in one short sentence and invite them to speak.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client) -> None:
        logger.info(f"Peer disconnected pc_id={webrtc_connection.pc_id}; cancelling task")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception:
        logger.exception("Bot pipeline crashed")
    finally:
        logger.info(f"Bot finished for pc_id={webrtc_connection.pc_id}")
