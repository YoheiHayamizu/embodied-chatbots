"""Unit tests for persona merging, strategy selection, and directive rendering.

These tests cover pure-functional behaviour: no pipecat, no anthropic SDK,
no I/O. They lock in the invariants that the rest of the SSLG pipeline
relies on (trait clamping, merge precedence, deterministic selection when
seeded, no strategy-label leakage in the directive text).
"""

from __future__ import annotations

import pytest

from reoh_bot.persona import (
    DEFAULT_STRATEGY_WEIGHTS,
    DIALOG_ACT_TO_STRATEGIES,
    DIALOG_STRATEGIES,
    EMPTY_PERSONA,
    Persona,
    StrategySelector,
    merged_with,
    render_directive,
)


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


def test_empty_persona_is_all_blank_with_neutral_traits() -> None:
    assert EMPTY_PERSONA.is_empty()
    assert EMPTY_PERSONA.extraversion == 3
    assert EMPTY_PERSONA.agreeableness == 3
    assert EMPTY_PERSONA.conscientiousness == 3


def test_persona_clamps_traits_to_valid_range() -> None:
    p = Persona(extraversion=99, agreeableness=-10, conscientiousness=3)
    assert p.extraversion == 5
    assert p.agreeableness == 1
    assert p.conscientiousness == 3


def test_persona_from_dict_tolerates_missing_and_malformed_fields() -> None:
    p = Persona.from_dict(
        {
            "age": "28",
            "interests": None,
            "extraversion": "4.6",
            "agreeableness": "not a number",
        }
    )
    assert p.age == "28"
    assert p.interests == ""
    assert p.extraversion == 5  # 4.6 rounds up, clamped to 5 is the same
    assert p.agreeableness == 3  # fallback default


def test_persona_merge_prefers_new_text_and_averages_traits() -> None:
    old = Persona(age="28", interests="drawing", extraversion=2)
    new = Persona(age="", occupation="illustrator", interests="painting", extraversion=4)

    merged = old.merge(new)

    # Text: new wins when non-empty, otherwise keep old.
    assert merged.age == "28"
    assert merged.occupation == "illustrator"
    assert merged.interests == "painting"
    # Trait: average of 2 and 4 = 3.
    assert merged.extraversion == 3


def test_persona_merge_keeps_old_when_new_is_empty_persona() -> None:
    old = Persona(age="28", extraversion=4)
    merged = old.merge(EMPTY_PERSONA)
    assert merged.age == "28"
    # Averaging with the default (3) nudges 4 back down to 3.5 -> 4 (rounded).
    assert merged.extraversion == 4


def test_persona_summary_empty_is_placeholder_not_blank() -> None:
    # Directive rendering must never emit an empty summary — the LLM needs
    # something to react to on the very first turn.
    text = EMPTY_PERSONA.summary()
    assert "no specific details" in text


def test_persona_summary_includes_provided_fields_and_traits() -> None:
    p = Persona(
        age="35",
        occupation="engineer",
        interests="cooking",
        extraversion=4,
        agreeableness=2,
        conscientiousness=5,
    )
    text = p.summary()
    assert "35" in text
    assert "engineer" in text
    assert "cooking" in text
    assert "4/2/5" in text


def test_persona_to_dict_roundtrip() -> None:
    p = Persona(age="40", interests="cycling", extraversion=5)
    assert Persona.from_dict(p.to_dict()) == p


def test_merged_with_none_returns_current_unchanged() -> None:
    current = Persona(age="28")
    assert merged_with(current, None) is current


# ---------------------------------------------------------------------------
# StrategySelector
# ---------------------------------------------------------------------------


def test_selector_returns_empty_for_unknown_act() -> None:
    selector = StrategySelector(seed=0)
    assert selector.select(EMPTY_PERSONA, dialog_act="chitchat") == ""


def test_selector_picks_from_inform_strategies() -> None:
    selector = StrategySelector(seed=42)
    choice = selector.select(EMPTY_PERSONA, dialog_act="inform")
    assert choice in DIALOG_ACT_TO_STRATEGIES["inform"]


def test_selector_picks_from_request_strategies() -> None:
    selector = StrategySelector(seed=42)
    choice = selector.select(EMPTY_PERSONA, dialog_act="request")
    assert choice in DIALOG_ACT_TO_STRATEGIES["request"]


def test_selector_is_deterministic_when_seeded() -> None:
    a = StrategySelector(seed=123)
    b = StrategySelector(seed=123)
    sequence_a = [a.select(EMPTY_PERSONA, dialog_act="inform") for _ in range(8)]
    sequence_b = [b.select(EMPTY_PERSONA, dialog_act="inform") for _ in range(8)]
    assert sequence_a == sequence_b


def test_extraversion_bump_shifts_distribution_toward_emotional_strategies() -> None:
    # Draw many samples with a neutral persona and a highly-extraverted one.
    # The extraverted persona should land on ``emotional_appeal`` or
    # ``personal_story`` more often.
    trials = 2000
    target = {"emotional_appeal", "personal_story"}

    neutral = StrategySelector(seed=1)
    extraverted = StrategySelector(seed=1)
    neutral_persona = EMPTY_PERSONA
    extraverted_persona = Persona(extraversion=5)

    neutral_hits = sum(
        1
        for _ in range(trials)
        if neutral.select(neutral_persona, dialog_act="inform") in target
    )
    extraverted_hits = sum(
        1
        for _ in range(trials)
        if extraverted.select(extraverted_persona, dialog_act="inform") in target
    )

    assert extraverted_hits > neutral_hits


def test_agreeableness_bump_favours_warm_strategies() -> None:
    trials = 2000
    target = {"self_modeling", "comment_partner"}

    neutral = StrategySelector(seed=7)
    agreeable = StrategySelector(seed=7)

    neutral_hits = sum(
        1
        for _ in range(trials)
        if neutral.select(EMPTY_PERSONA, dialog_act="inform") in target
    )
    agreeable_hits = sum(
        1
        for _ in range(trials)
        if agreeable.select(Persona(agreeableness=5), dialog_act="inform") in target
    )

    assert agreeable_hits > neutral_hits


def test_selector_zero_weights_falls_back_to_uniform_choice() -> None:
    zero_weights = {name: 0.0 for name in DEFAULT_STRATEGY_WEIGHTS}
    selector = StrategySelector(weights=zero_weights, seed=0)
    choice = selector.select(EMPTY_PERSONA, dialog_act="inform")
    assert choice in DIALOG_ACT_TO_STRATEGIES["inform"]


# ---------------------------------------------------------------------------
# render_directive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", sorted(DIALOG_STRATEGIES.values()))
def test_render_directive_does_not_leak_strategy_identifier(strategy: str) -> None:
    # The label (e.g. ``emotional_appeal``) must not appear in the text the
    # LLM sees. Each strategy identifier has an underscore that the
    # human-facing description does not.
    directive = render_directive(Persona(age="30"), strategy)
    assert directive["role"] == "developer"
    assert strategy not in directive["content"]


def test_render_directive_includes_persona_summary_and_room() -> None:
    directive = render_directive(
        Persona(age="28", occupation="illustrator", extraversion=4),
        "emotional_appeal",
        current_room="livingroom",
    )
    content = directive["content"]
    assert "28" in content
    assert "illustrator" in content
    assert "livingroom" in content


def test_render_directive_with_empty_persona_still_well_formed() -> None:
    directive = render_directive(EMPTY_PERSONA, "logical_appeal")
    assert "no specific details" in directive["content"]


def test_render_directive_unknown_strategy_uses_neutral_fallback() -> None:
    directive = render_directive(EMPTY_PERSONA, "nonexistent_strategy")
    assert "respond naturally" in directive["content"]
