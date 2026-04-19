"""Unit tests for :class:`reoh_bot.persona_processor.PersonaStrategyProcessor`.

These tests exercise the processor's contract without booting a full
pipecat pipeline: we feed it ``LLMRunFrame`` directly, assert that a
directive was appended to the context before the frame is forwarded, and
cover the failure and empty-context paths.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import LLMRunFrame, TextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from reoh_bot.persona import EMPTY_PERSONA, Persona, StrategySelector
from reoh_bot.persona_extractor import PersonaExtractor, PersonaExtractorSettings
from reoh_bot.persona_processor import PersonaStrategyProcessor


def _build_processor(
    *,
    extractor: PersonaExtractor | None = None,
    dialog_act: str = "inform",
) -> tuple[PersonaStrategyProcessor, LLMContext]:
    context = LLMContext()
    processor = PersonaStrategyProcessor(
        context=context,
        extractor=extractor,
        selector=StrategySelector(seed=0),
        dialog_act=dialog_act,
    )
    return processor, context


def _drain_pushed(processor: PersonaStrategyProcessor) -> list[Any]:
    """Record frames pushed downstream by replacing ``push_frame``."""
    pushed: list[Any] = []

    async def _fake_push(frame: Any, direction: FrameDirection) -> None:
        pushed.append((frame, direction))

    processor.push_frame = _fake_push  # type: ignore[assignment]
    return pushed


def test_processor_injects_directive_before_forwarding_llm_run_frame() -> None:
    processor, context = _build_processor()
    context.add_message({"role": "user", "content": "I work as a freelance illustrator."})
    pushed = _drain_pushed(processor)

    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    # Directive was appended after the user message.
    assert len(context.messages) == 2
    directive = context.messages[-1]
    assert directive["role"] == "developer"
    # Frame was still forwarded.
    assert len(pushed) == 1
    assert isinstance(pushed[0][0], LLMRunFrame)


def test_processor_skips_directive_when_context_has_no_user_message() -> None:
    processor, context = _build_processor()
    pushed = _drain_pushed(processor)

    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    # No user message means no directive (the opening greeting turn).
    assert context.messages == []
    assert len(pushed) == 1


def test_processor_passes_through_non_run_frames_without_side_effects() -> None:
    processor, context = _build_processor()
    context.add_message({"role": "user", "content": "hello there"})
    pushed = _drain_pushed(processor)

    asyncio.run(
        processor.process_frame(TextFrame(text="unused"), FrameDirection.DOWNSTREAM)
    )

    # Context is unchanged — only LLMRunFrame triggers injection.
    assert len(context.messages) == 1
    assert pushed[0][0].text == "unused"


def test_processor_skips_directive_on_upstream_run_frame() -> None:
    processor, context = _build_processor()
    context.add_message({"role": "user", "content": "I love cycling on weekends."})
    pushed = _drain_pushed(processor)

    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.UPSTREAM))

    assert len(context.messages) == 1  # no directive added
    assert pushed[0][1] == FrameDirection.UPSTREAM


def test_processor_updates_persona_from_extractor() -> None:
    payload = json.dumps(
        {
            "age": "",
            "occupation": "illustrator",
            "family": "",
            "interests": "",
            "lifestyle": "",
            "extraversion": 4,
            "agreeableness": 3,
            "conscientiousness": 3,
        }
    )

    class _Stub:
        async def create(self, **kwargs: Any) -> Any:
            class _Block:
                text = payload

            class _Resp:
                content = [_Block()]

            return _Resp()

    class _Client:
        messages = _Stub()

    extractor = PersonaExtractor(
        settings=PersonaExtractorSettings(
            api_key="x", timeout_s=2.0, min_utterance_tokens=3
        ),
        system_prompt="sys",
        client=_Client(),
    )

    processor, context = _build_processor(extractor=extractor)
    context.add_message({"role": "user", "content": "I am a freelance illustrator."})
    _drain_pushed(processor)

    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    assert processor.persona.occupation == "illustrator"
    assert processor.persona.extraversion >= 3


def test_processor_survives_extractor_exception() -> None:
    mock = AsyncMock(side_effect=RuntimeError("boom"))

    class _StubExtractor:
        extract = mock

    processor, context = _build_processor(extractor=_StubExtractor())  # type: ignore[arg-type]
    context.add_message({"role": "user", "content": "I work from home daily."})
    pushed = _drain_pushed(processor)

    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    # Persona stays at default, frame still forwarded, directive still added
    # so the LLM has some persona context to anchor on.
    assert processor.persona == EMPTY_PERSONA
    assert context.messages[-1]["role"] == "developer"
    assert len(pushed) == 1


@pytest.mark.parametrize(
    "content_form",
    [
        "plain string content",
        [{"type": "text", "text": "block-shaped content"}],
    ],
)
def test_processor_reads_user_text_from_either_content_shape(content_form: Any) -> None:
    processor, context = _build_processor()
    context.add_message({"role": "user", "content": content_form})
    _drain_pushed(processor)

    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    directive_content = context.messages[-1]["content"]
    # Persona is still empty so the summary placeholder appears; that's
    # fine — what matters is the directive got injected at all.
    assert "For this reply" in directive_content


def test_processor_reuses_persona_across_turns() -> None:
    """Second turn must build on the first turn's persona, not reset it."""
    payload_turn1 = json.dumps(
        Persona(occupation="illustrator").to_dict()
    )
    payload_turn2 = json.dumps(
        Persona(interests="cafe hopping").to_dict()
    )
    payloads = iter([payload_turn1, payload_turn2])

    class _Stub:
        async def create(self, **kwargs: Any) -> Any:
            text = next(payloads)

            class _Block:
                pass

            block = _Block()
            block.text = text  # type: ignore[attr-defined]

            class _Resp:
                content = [block]

            return _Resp()

    class _Client:
        messages = _Stub()

    extractor = PersonaExtractor(
        settings=PersonaExtractorSettings(api_key="x", min_utterance_tokens=3),
        system_prompt="sys",
        client=_Client(),
    )

    processor, context = _build_processor(extractor=extractor)
    _drain_pushed(processor)

    context.add_message({"role": "user", "content": "I am an illustrator."})
    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    context.add_message({"role": "user", "content": "I enjoy cafe hopping too."})
    asyncio.run(processor.process_frame(LLMRunFrame(), FrameDirection.DOWNSTREAM))

    assert processor.persona.occupation == "illustrator"
    assert processor.persona.interests == "cafe hopping"
