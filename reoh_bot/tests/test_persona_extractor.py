"""Unit tests for :mod:`reoh_bot.persona_extractor`.

The real extractor makes network calls to Anthropic, which is out of
scope for unit tests. Instead we inject a stub client that mimics the
shape of :class:`anthropic.AsyncAnthropic` — just the one coroutine
``client.messages.create`` that returns an object with a ``content`` list
of text blocks.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from reoh_bot.persona import EMPTY_PERSONA, Persona
from reoh_bot.persona_extractor import (
    PersonaExtractor,
    PersonaExtractorSettings,
    extract_or_keep,
)


# ---------------------------------------------------------------------------
# Stubs mimicking the anthropic SDK's response shape
# ---------------------------------------------------------------------------


@dataclass
class _TextBlock:
    text: str


@dataclass
class _Response:
    content: list[_TextBlock]


class _StubMessages:
    def __init__(self, payload: str | Exception, delay: float = 0.0) -> None:
        self.payload = payload
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.payload, Exception):
            raise self.payload
        return _Response(content=[_TextBlock(text=self.payload)])


class _StubClient:
    def __init__(self, payload: str | Exception, delay: float = 0.0) -> None:
        self.messages = _StubMessages(payload, delay=delay)


def _settings(**overrides: Any) -> PersonaExtractorSettings:
    base = dict(
        api_key="test",
        model="claude-haiku-4-5",
        max_tokens=256,
        timeout_s=2.0,
        min_utterance_tokens=3,
    )
    base.update(overrides)
    return PersonaExtractorSettings(**base)


def _extractor(payload: str | Exception, delay: float = 0.0, **overrides: Any) -> PersonaExtractor:
    return PersonaExtractor(
        settings=_settings(**overrides),
        system_prompt="sys",
        client=_StubClient(payload, delay=delay),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_parses_bare_json_object() -> None:
    payload = json.dumps(
        {
            "age": "I am 30 years old",
            "occupation": "illustrator",
            "family": "",
            "interests": "",
            "lifestyle": "",
            "extraversion": 4,
            "agreeableness": 3,
            "conscientiousness": 3,
        }
    )
    extractor = _extractor(payload)

    result = asyncio.run(
        extractor.extract("I'm thirty and I work as an illustrator.", EMPTY_PERSONA)
    )

    assert isinstance(result, Persona)
    assert result.age == "I am 30 years old"
    assert result.occupation == "illustrator"
    assert result.extraversion == 4


def test_extract_skips_utterances_below_min_tokens() -> None:
    extractor = _extractor("should not be called")
    result = asyncio.run(extractor.extract("ok", EMPTY_PERSONA))
    assert result is None
    # No call made at all — early return before the Claude round-trip.
    assert extractor.client.messages.calls == []


def test_extract_handles_unparseable_output() -> None:
    extractor = _extractor("I don't feel like returning JSON today.")
    result = asyncio.run(
        extractor.extract("I love cooking on weekends", EMPTY_PERSONA)
    )
    assert result is None


def test_extract_tolerates_json_wrapped_in_prose() -> None:
    wrapped = (
        "Sure — here is the delta:\n"
        '{"age":"","occupation":"","family":"","interests":"cycling",'
        '"lifestyle":"","extraversion":3,"agreeableness":3,"conscientiousness":3}\n'
        "Let me know if you need anything else."
    )
    extractor = _extractor(wrapped)
    result = asyncio.run(
        extractor.extract("I ride my bike every morning.", EMPTY_PERSONA)
    )
    assert result is not None
    assert result.interests == "cycling"


def test_extract_returns_none_on_timeout() -> None:
    # Model takes 1s but timeout is 0.05s.
    extractor = _extractor('{"age":"30"}', delay=1.0, timeout_s=0.05)
    result = asyncio.run(
        extractor.extract("I am thirty years old.", EMPTY_PERSONA)
    )
    assert result is None


def test_extract_returns_none_on_client_exception() -> None:
    extractor = _extractor(RuntimeError("network down"))
    result = asyncio.run(
        extractor.extract("I am a part-time musician.", EMPTY_PERSONA)
    )
    assert result is None


def test_extract_sends_prior_and_utterance_in_user_message() -> None:
    payload = json.dumps(Persona().to_dict())
    extractor = _extractor(payload)
    prior = Persona(age="28", occupation="illustrator")

    asyncio.run(extractor.extract("I also enjoy cafe hopping.", prior))

    call = extractor.client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == "sys"
    user_content = call["messages"][0]["content"]
    assert "PRIOR:" in user_content
    assert "illustrator" in user_content
    assert "cafe hopping" in user_content


def test_extract_or_keep_returns_prior_when_extractor_is_none() -> None:
    prior = Persona(age="28")
    result = asyncio.run(extract_or_keep(None, "hello", prior))
    assert result is prior


def test_extract_or_keep_merges_delta_into_prior() -> None:
    payload = json.dumps(
        {
            "age": "",
            "occupation": "illustrator",
            "family": "",
            "interests": "",
            "lifestyle": "",
            "extraversion": 3,
            "agreeableness": 3,
            "conscientiousness": 3,
        }
    )
    extractor = _extractor(payload)
    prior = Persona(age="28")

    result = asyncio.run(
        extract_or_keep(extractor, "I work as an illustrator.", prior)
    )

    assert result.age == "28"
    assert result.occupation == "illustrator"


def test_extract_or_keep_returns_prior_when_delta_is_none() -> None:
    extractor = _extractor("not json")
    prior = Persona(age="28")
    result = asyncio.run(
        extract_or_keep(extractor, "long enough utterance here", prior)
    )
    assert result == prior


def test_extract_handles_empty_response_body() -> None:
    # Anthropic occasionally returns an empty content list when the model
    # refuses to produce output — the extractor should not crash.
    client = _StubClient("does not matter")
    client.messages = _StubMessages("")  # type: ignore[assignment]

    extractor = PersonaExtractor(settings=_settings(), system_prompt="sys", client=client)
    result = asyncio.run(
        extractor.extract("I love cooking", EMPTY_PERSONA)
    )
    assert result is None


@pytest.mark.parametrize("utterance", ["", "  ", "\n"])
def test_extract_skips_blank_utterances(utterance: str) -> None:
    extractor = _extractor("irrelevant")
    result = asyncio.run(extractor.extract(utterance, EMPTY_PERSONA))
    assert result is None
