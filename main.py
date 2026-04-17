import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
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
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    # Local STT（Faster Whisper）
    stt = WhisperSTTService(
        device="auto",              # Use GPU if available, otherwise CPU
        compute_type="int8",        # Use int8 quantized model for faster inference with minimal accuracy loss
        settings=WhisperSTTService.Settings(
            model="large-v3-turbo",       # Use the base model for a good balance of speed and accuracy
            language=Language.EN,
        ),
    )

    # Local TTS（Piper）
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice="en_US-ryan-high"),
        download_dir=Path("./models/piper"),
    )

    # Cloud LLM
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        settings=AnthropicLLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Responses will be spoken aloud—avoid emojis or bullet lists."
            ),
        ),
    )

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
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    context.add_message(
        {"role": "developer", "content": "Please introduce yourself to the user."}
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
