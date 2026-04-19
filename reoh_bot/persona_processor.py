"""Pipecat frame processor that adds persona-aware strategy directives.

The SSLG pipeline looks like::

    transport.input() -> stt -> user_aggregator -> PersonaStrategyProcessor
                                               -> agent.llm -> tts -> transport.output()

The user-aggregator appends the visitor's latest transcript to the shared
:class:`LLMContext` and then emits an :class:`LLMRunFrame` downstream to
trigger generation. This processor intercepts that run-frame, reads the
newly-added user message out of the context, runs the persona extractor
synchronously, picks a strategy, appends a per-turn developer-role
directive to the context, and only then forwards the run-frame to the LLM.

Because the extractor is awaited inline, the total turn latency grows by
~Haiku-call time. The design choice to accept this (rather than
fire-and-forget asynchronous extraction) is intentional: running the
directive through the LLM on the *same* turn that the utterance arrived
is a much simpler mental model than "persona always lags one turn
behind", and Haiku's latency (≈300–800 ms) is within the voice bot's
tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from loguru import logger

from pipecat.frames.frames import Frame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from reoh_bot.persona import (
    EMPTY_PERSONA,
    Persona,
    StrategySelector,
    render_directive,
)
from reoh_bot.persona_extractor import PersonaExtractor, extract_or_keep


def _latest_user_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """Return the text of the most recent user-role message in ``messages``.

    The Anthropic message shape allows ``content`` to be either a string or
    a list of content blocks. We handle both and fall back to an empty
    string if there is no user message yet.
    """
    for msg in reversed(list(messages)):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(part for part in parts if part)
        return ""
    return ""


@dataclass
class PersonaStrategyProcessor(FrameProcessor):
    """Inject a persona+strategy directive immediately before each LLM run.

    Construct with the shared :class:`LLMContext`, the configured
    :class:`PersonaExtractor` (``None`` disables extraction), and a
    :class:`StrategySelector`. The processor keeps a single mutable
    :class:`Persona` snapshot in memory — reset on each new pipeline
    instance since pipelines are one-per-visit.
    """

    context: LLMContext
    extractor: PersonaExtractor | None
    selector: StrategySelector
    dialog_act: str = "inform"
    _persona: Persona = field(default=EMPTY_PERSONA, init=False, repr=False)

    def __post_init__(self) -> None:
        # ``FrameProcessor.__init__`` sets up queues and the like; dataclass
        # doesn't call it automatically, so we invoke it explicitly here.
        super().__init__()

    @property
    def persona(self) -> Persona:
        """Current merged persona snapshot (exposed for tests and logging)."""
        return self._persona

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Always let the base class track StartFrame/CancelFrame/etc.
        await super().process_frame(frame, direction)

        # Only downstream LLMRunFrames are interesting: they mark the
        # moment the user-aggregator has finished stitching the latest
        # turn onto the context and wants the LLM to reply.
        if isinstance(frame, LLMRunFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._inject_directive()

        await self.push_frame(frame, direction)

    async def _inject_directive(self) -> None:
        """Extract persona from the latest user message and append a directive."""
        utterance = _latest_user_text(self.context.get_messages())
        if not utterance:
            # No user turn yet (e.g. opening-directive run). Skip persona
            # work but still let the frame through — otherwise the greeting
            # turn never runs.
            return

        try:
            self._persona = await extract_or_keep(
                self.extractor, utterance, self._persona
            )
        except Exception:  # noqa: BLE001 — extraction must never break the pipeline
            logger.exception("persona-processor: extraction raised; keeping prior")

        strategy = self.selector.select(self._persona, dialog_act=self.dialog_act)
        directive = render_directive(self._persona, strategy)
        self.context.add_message(directive)
        logger.debug(
            "persona-processor: injected directive strategy={} persona={}",
            strategy or "none",
            self._persona.summary(),
        )


__all__ = ["PersonaStrategyProcessor"]
