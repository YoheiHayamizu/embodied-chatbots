"""Persona tracking and strategy selection for the SSLG agent.

This module is a clean, self-contained port of the strategy layer in the
upstream ``reoh`` project (``reoh.core.constants`` +
``reoh.pipeline.policy.template_policy.StrategySelector``) tailored to a
per-turn, in-memory use case inside a pipecat voice pipeline.

Three concerns live here, intentionally co-located because they are small
and always used together:

* :class:`Persona` — an immutable snapshot of what we have inferred about
  the visitor so far, with a :meth:`Persona.merge` operator that combines an
  old snapshot with a newly extracted one. The merge keeps old text values
  unless the new one is non-empty, and averages the Big-Five trait scores
  so a single outlier utterance cannot swing the persona wildly.
* :class:`StrategySelector` — a seedable, weighted picker that chooses a
  communication strategy (``logical_appeal``, ``emotional_appeal``, etc.)
  based on the current persona's traits and the dialog act.
* :func:`render_directive` — turns a persona + strategy pair into a
  ``developer``-role message suitable for appending to the pipecat
  ``LLMContext`` right before the main LLM runs. The directive phrases the
  strategy in natural language so the identifier (``emotional_appeal``)
  never leaks into the spoken reply.

The module has no pipecat or anthropic dependencies so it can be unit-tested
in isolation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Mapping


# ---------------------------------------------------------------------------
# Strategy constants (mirrored from reoh.core.constants)
# ---------------------------------------------------------------------------

DIALOG_STRATEGIES: Mapping[str, str] = {
    "LOGICAL_APPEAL": "logical_appeal",
    "EMOTIONAL_APPEAL": "emotional_appeal",
    "PERSONAL_STORY": "personal_story",
    "HIGHLIGHT_UNIQUE": "highlight_unique_feature",
    "SELF_MODELING": "self_modeling",
    "COMMENT_PARTNER": "comment_partner",
    "PERSONAL_INQUIRY": "personal_related_inquiry",
    "INTEREST_INQUIRY": "interest_level_inquiry",
}

# Which strategies are valid for which dialog act. ``inform`` strategies
# persuade or describe; ``request`` strategies probe the visitor.
DIALOG_ACT_TO_STRATEGIES: Mapping[str, tuple[str, ...]] = {
    "inform": (
        "logical_appeal",
        "emotional_appeal",
        "personal_story",
        "highlight_unique_feature",
        "self_modeling",
        "comment_partner",
    ),
    "request": (
        "personal_related_inquiry",
        "interest_level_inquiry",
    ),
}

DEFAULT_STRATEGY_WEIGHTS: Mapping[str, float] = {
    "logical_appeal": 0.30,
    "emotional_appeal": 0.30,
    "personal_story": 0.10,
    "highlight_unique_feature": 0.20,
    "self_modeling": 0.05,
    "comment_partner": 0.05,
    "personal_related_inquiry": 0.50,
    "interest_level_inquiry": 0.50,
}

# Human-readable descriptions injected into the per-turn directive so the
# main LLM knows *how* to colour the reply without ever seeing the
# strategy's machine identifier.
STRATEGY_DESCRIPTIONS: Mapping[str, str] = {
    "logical_appeal": (
        "lead with facts and practical reasoning — point out the concrete "
        "benefit of the feature in plain, non-salesy language"
    ),
    "emotional_appeal": (
        "evoke the feeling of living with this feature — paint a brief "
        "sensory picture without exaggeration or marketing speak"
    ),
    "personal_story": (
        "share a short anecdote about a past visitor or client who "
        "enjoyed this kind of feature, framed as a natural aside"
    ),
    "highlight_unique_feature": (
        "emphasise what makes this feature distinctive compared with "
        "typical homes, factually and without superlatives"
    ),
    "self_modeling": (
        "express your own mild preference or enjoyment of the feature "
        "(\"I find it works well for…\") to lead by example"
    ),
    "comment_partner": (
        "make a brief, friendly aside that keeps the conversation warm "
        "without pushing the feature"
    ),
    "personal_related_inquiry": (
        "ask a gentle personal question (lifestyle, family, hobbies) that "
        "helps you understand how this home would fit the visitor"
    ),
    "interest_level_inquiry": (
        "ask how the visitor feels about the current room or feature, to "
        "gauge their interest before moving on"
    ),
}


# ---------------------------------------------------------------------------
# Persona snapshot
# ---------------------------------------------------------------------------


def _clamp_trait(value: float) -> int:
    """Clamp a Big-Five trait to the 1..5 integer range."""
    return max(1, min(5, int(round(value))))


@dataclass(frozen=True)
class Persona:
    """Immutable snapshot of the visitor's inferred persona.

    Text fields mirror the upstream ``reoh`` persona schema so scenario JSONs
    can be reused verbatim. Integer trait scores (Big-Five subset) are kept
    on a 1..5 Likert scale and are clamped on construction.

    Use :meth:`merge` to combine a prior snapshot with a freshly extracted
    one: non-empty text replaces empty text, and trait scores are averaged
    so extraction noise is damped.
    """

    age: str = ""
    occupation: str = ""
    family: str = ""
    interests: str = ""
    lifestyle: str = ""
    extraversion: int = 3
    agreeableness: int = 3
    conscientiousness: int = 3

    def __post_init__(self) -> None:
        # ``frozen=True`` forbids normal attribute assignment, but
        # ``object.__setattr__`` is the accepted escape hatch for post-init
        # validation of a frozen dataclass.
        object.__setattr__(self, "extraversion", _clamp_trait(self.extraversion))
        object.__setattr__(self, "agreeableness", _clamp_trait(self.agreeableness))
        object.__setattr__(self, "conscientiousness", _clamp_trait(self.conscientiousness))

    def merge(self, other: "Persona") -> "Persona":
        """Return a new persona combining ``self`` with a newer ``other``.

        Text fields: ``other`` wins when non-empty, otherwise keep ``self``.
        Trait fields: arithmetic mean (rounded, clamped to 1..5).
        """
        return Persona(
            age=other.age or self.age,
            occupation=other.occupation or self.occupation,
            family=other.family or self.family,
            interests=other.interests or self.interests,
            lifestyle=other.lifestyle or self.lifestyle,
            extraversion=_clamp_trait((self.extraversion + other.extraversion) / 2),
            agreeableness=_clamp_trait((self.agreeableness + other.agreeableness) / 2),
            conscientiousness=_clamp_trait(
                (self.conscientiousness + other.conscientiousness) / 2
            ),
        )

    def is_empty(self) -> bool:
        """``True`` if no text fields have been observed yet."""
        return not any(
            (self.age, self.occupation, self.family, self.interests, self.lifestyle)
        )

    def summary(self) -> str:
        """One-line natural-language summary for the directive message.

        Empty persona returns a short placeholder so the prompt stays
        well-formed on the very first turn.
        """
        if self.is_empty():
            return (
                "no specific details yet (traits default to neutral mid-scale values)"
            )
        parts: list[str] = []
        if self.age:
            parts.append(self.age)
        if self.occupation:
            parts.append(self.occupation)
        if self.family:
            parts.append(self.family)
        if self.interests:
            parts.append(f"interests: {self.interests}")
        if self.lifestyle:
            parts.append(f"lifestyle: {self.lifestyle}")
        parts.append(
            f"traits E/A/C={self.extraversion}/{self.agreeableness}/{self.conscientiousness}"
        )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "age": self.age,
            "occupation": self.occupation,
            "family": self.family,
            "interests": self.interests,
            "lifestyle": self.lifestyle,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "conscientiousness": self.conscientiousness,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Persona":
        """Build a persona from a JSON-like mapping.

        Missing fields fall back to the dataclass defaults so a partial
        extraction still yields a valid :class:`Persona`.
        """

        def _text(key: str) -> str:
            value = data.get(key, "")
            return str(value) if value is not None else ""

        def _trait(key: str, default: int) -> int:
            raw = data.get(key, default)
            try:
                return _clamp_trait(float(raw))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        return cls(
            age=_text("age"),
            occupation=_text("occupation"),
            family=_text("family"),
            interests=_text("interests"),
            lifestyle=_text("lifestyle"),
            extraversion=_trait("extraversion", 3),
            agreeableness=_trait("agreeableness", 3),
            conscientiousness=_trait("conscientiousness", 3),
        )


EMPTY_PERSONA = Persona()


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


@dataclass
class StrategySelector:
    """Weighted, persona-aware strategy picker.

    Construct once per session and reuse across turns. The selector owns an
    internal :class:`random.Random` so call sites can pass a seed for
    deterministic tests without touching the global RNG.
    """

    weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STRATEGY_WEIGHTS)
    )
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(
        self,
        persona: Persona,
        *,
        dialog_act: str = "inform",
        current_room: str | None = None,
    ) -> str:
        """Pick a strategy for the upcoming turn.

        Returns an empty string when ``dialog_act`` has no associated
        strategies (so callers can skip directive injection without special
        error handling).
        """
        candidates = DIALOG_ACT_TO_STRATEGIES.get(dialog_act, ())
        if not candidates:
            return ""

        # Start from the configured base weights, dropping strategies that
        # are not valid for the requested act.
        adjusted: dict[str, float] = {
            name: float(self.weights.get(name, 0.0)) for name in candidates
        }

        # Persona-trait bumps mirror the heuristic in the upstream
        # ``StrategySelector``. Values are multiplicative so neutral traits
        # (3) leave weights unchanged.
        if persona.extraversion > 3:
            if "personal_story" in adjusted:
                adjusted["personal_story"] *= 1.5
            if "emotional_appeal" in adjusted:
                adjusted["emotional_appeal"] *= 1.2

        if persona.agreeableness > 3:
            if "self_modeling" in adjusted:
                adjusted["self_modeling"] *= 1.2
            if "comment_partner" in adjusted:
                adjusted["comment_partner"] *= 1.5

        if persona.conscientiousness > 3:
            if "logical_appeal" in adjusted:
                adjusted["logical_appeal"] *= 1.3

        # Room-level bumps keep the heuristic: emotional tone plays better in
        # living spaces, personal stories around food.
        if current_room in {"livingroom", "bedroom"} and "emotional_appeal" in adjusted:
            adjusted["emotional_appeal"] *= 1.2
        if current_room == "kitchen" and "personal_story" in adjusted:
            adjusted["personal_story"] *= 1.3

        total = sum(adjusted.values())
        if total <= 0:
            return self._rng.choice(candidates)

        return self._rng.choices(
            population=list(adjusted.keys()),
            weights=list(adjusted.values()),
            k=1,
        )[0]


# ---------------------------------------------------------------------------
# Directive rendering
# ---------------------------------------------------------------------------


def render_directive(
    persona: Persona,
    strategy: str,
    *,
    current_room: str | None = None,
) -> Mapping[str, str]:
    """Build the per-turn ``developer``-role message for the LLM context.

    The message must never name the ``strategy`` identifier directly — the
    LLM is instructed (in the system prompt) not to echo it, but we also
    don't feed it the label in the first place: only the natural-language
    description is included.
    """
    description = STRATEGY_DESCRIPTIONS.get(strategy, "")
    if not description:
        # Unknown or empty strategy: emit a soft directive so the context
        # still has a persona summary for the model to react to.
        description = "respond naturally without a specific persuasive angle"

    room_clause = f" (current room: {current_room})" if current_room else ""
    content = (
        f"Visitor so far{room_clause}: {persona.summary()}. "
        f"For this reply, {description}. "
        "Keep it to one or two short sentences and do not mention that you are "
        "adapting your style."
    )
    return {"role": "developer", "content": content}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def merged_with(current: Persona, updated: Persona | None) -> Persona:
    """Thin wrapper around :meth:`Persona.merge` that tolerates ``None``.

    Handy for extractor call-sites where the new snapshot may be ``None`` on
    parse failure and the caller just wants to keep the prior one.
    """
    if updated is None:
        return current
    return current.merge(updated)


__all__ = [
    "DEFAULT_STRATEGY_WEIGHTS",
    "DIALOG_ACT_TO_STRATEGIES",
    "DIALOG_STRATEGIES",
    "EMPTY_PERSONA",
    "Persona",
    "STRATEGY_DESCRIPTIONS",
    "StrategySelector",
    "merged_with",
    "render_directive",
]


# ``replace`` is re-exported for tests that want to tweak a single field of
# a persona without rebuilding the whole dataclass.
_ = replace  # noqa: F841 — re-export surface
