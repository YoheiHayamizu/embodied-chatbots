"""Synchronous per-turn persona extractor backed by Claude Haiku.

Each user utterance is passed through a lightweight Anthropic call that
returns a JSON persona delta. Deltas are merged into a running
:class:`reoh_bot.persona.Persona` by the caller (see
:mod:`reoh_bot.persona_processor`), so this module itself stays stateless.

Design notes:

* The extractor uses :class:`anthropic.AsyncAnthropic` directly rather than
  going through ``pipecat.services.anthropic.llm.AnthropicLLMService``.
  Pipecat's wrapper is built for streaming chat, which is overkill for a
  one-shot JSON call and adds frame-handling overhead to the hot path.
* Failure modes — timeout, parse error, empty body — all return ``None``
  so the caller can fall back to the prior persona without branching on
  error types. We log (never ``print``) the reason so issues are visible
  in ``logs/`` without leaking into the voice pipeline.
* The extractor prompt ships as a Markdown file alongside the other
  templates (``reoh_bot/prompts/persona_extractor_prompt.md``) so edits
  do not require a code change.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

try:  # pragma: no cover — import is trivial
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover — pipecat[anthropic] brings this in
    AsyncAnthropic = None  # type: ignore[assignment]

from reoh_bot.persona import EMPTY_PERSONA, Persona


@dataclass(frozen=True)
class PersonaExtractorSettings:
    """Anthropic configuration for the extractor sub-call.

    Defaults are tuned for low latency: Haiku, a small ``max_tokens`` cap
    (JSON objects don't need many tokens), and a 4-second timeout that is
    generous enough for a single Haiku call while still bounded so a slow
    API response can't stall the voice pipeline.
    """

    api_key: str
    model: str = "claude-haiku-4-5"
    max_tokens: int = 256
    timeout_s: float = 4.0
    min_utterance_tokens: int = 3


def load_extractor_prompt(path: Path) -> str:
    """Read the persona-extractor system prompt from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Persona extractor prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_user_message(prior: Persona, utterance: str) -> str:
    """Compose the user-role message shown to the extractor model."""
    prior_json = json.dumps(prior.to_dict(), ensure_ascii=False)
    return f"PRIOR:\n{prior_json}\n\nUTTERANCE:\n{utterance}"


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    """Best-effort JSON object extraction from the model's reply.

    The extractor prompt asks for bare JSON, but real outputs sometimes
    include stray whitespace, leading ``json``-fenced code blocks, or a
    sentence before the object. We scan for the first balanced
    ``{...}`` substring and attempt to parse that.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    return parsed


@dataclass
class PersonaExtractor:
    """Calls Claude to produce a persona delta for a single utterance.

    ``client`` is injectable (and defaults to a freshly-constructed
    :class:`AsyncAnthropic`) so tests can pass a stub without patching
    globals.
    """

    settings: PersonaExtractorSettings
    system_prompt: str
    client: Any = None

    def __post_init__(self) -> None:
        if self.client is None:
            if AsyncAnthropic is None:  # pragma: no cover — guarded at import
                raise RuntimeError(
                    "anthropic package is not available — install pipecat-ai[anthropic]"
                )
            self.client = AsyncAnthropic(api_key=self.settings.api_key)

    async def extract(self, utterance: str, prior: Persona) -> Persona | None:
        """Return a persona delta for ``utterance`` or ``None`` on failure.

        Short or empty utterances are skipped cheaply — the pipeline sees
        plenty of backchannel noise ("uh-huh", "okay") and spending a
        Claude call on each would blow the latency budget.
        """
        trimmed = utterance.strip()
        if len(trimmed.split()) < self.settings.min_utterance_tokens:
            return None

        user_message = _build_user_message(prior, trimmed)

        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.settings.model,
                    max_tokens=self.settings.max_tokens,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                ),
                timeout=self.settings.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "persona-extractor: timeout after {}s, keeping prior persona",
                self.settings.timeout_s,
            )
            return None
        except Exception:  # noqa: BLE001 — best-effort: never break the voice turn
            logger.exception("persona-extractor: Claude call failed")
            return None

        text = _first_text_block(response)
        if not text:
            logger.warning("persona-extractor: empty response from model")
            return None

        parsed = _extract_json_object(text)
        if parsed is None:
            logger.warning("persona-extractor: could not parse JSON from {!r}", text[:200])
            return None

        try:
            return Persona.from_dict(parsed)
        except Exception:  # noqa: BLE001 — parse-shape failure is non-fatal
            logger.exception("persona-extractor: Persona.from_dict failed on {!r}", parsed)
            return None


def _first_text_block(response: Any) -> str:
    """Extract the first text block from an Anthropic ``Message`` response.

    Anthropic's Python SDK returns a ``Message`` with a ``content`` list of
    block objects; for a vanilla JSON reply there is exactly one
    ``TextBlock``. We defensively handle the case where the list is empty
    or a different block type is returned.
    """
    content = getattr(response, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            return text
    return ""


async def extract_or_keep(
    extractor: PersonaExtractor | None,
    utterance: str,
    prior: Persona,
) -> Persona:
    """Run the extractor (if any) and merge the delta with ``prior``.

    When ``extractor`` is ``None`` (the persona feature is disabled) this
    function returns ``prior`` unchanged, letting callers stay linear
    without an ``if extractor is None`` branch at every use site.
    """
    if extractor is None:
        return prior
    delta = await extractor.extract(utterance, prior)
    if delta is None:
        return prior
    # An extraction of "no personal signal" returns EMPTY_PERSONA, which
    # would nudge trait averages back toward 3 — desirable damping.
    return prior.merge(delta)


__all__ = [
    "EMPTY_PERSONA",
    "PersonaExtractor",
    "PersonaExtractorSettings",
    "extract_or_keep",
    "load_extractor_prompt",
]
