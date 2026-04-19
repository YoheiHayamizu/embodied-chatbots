"""End-to-End Language Generation (E2LG) agent for REOH voice tours.

This module is a clean rewrite of the upstream ``reoh.agents.e2lg_agent.E2LGAgent``
focused on the voice-pipeline use case:

* The LLM is **Claude** (Anthropic), not OpenAI. The provider stays out of the
  agent surface — :class:`E2LGAgent` only knows about the
  ``pipecat.services.anthropic.llm.AnthropicLLMService`` it wraps.
* Prompt construction is a pure function of an immutable
  :class:`~reoh_bot.scenarios.Scenario` and a prompt template, so prompts are
  cheap to rebuild and trivial to test.
* Dialog history is owned by pipecat's ``LLMContextAggregatorPair``; we do not
  re-implement it here. Anthropic's prompt caching makes the static system
  instruction effectively free per turn.
* The agent exposes its underlying pipecat service (``E2LGAgent.llm``) so it
  can drop straight into a :class:`pipecat.pipeline.pipeline.Pipeline`.

Compared to the original ``reoh`` implementation this drops:

* the ``OpenAI`` client, the unused ``State`` mutation, and the ``model.history``
  list that grew unbounded across turns;
* the ``print()`` debug statements and the ``temperature`` typo
  (``tempature``) that meant the value was never actually applied;
* the implicit dependency on ``self.property`` being non-None at construction
  time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from loguru import logger
from pipecat.services.anthropic.llm import AnthropicLLMService

from reoh_bot.scenarios import Scenario


@dataclass(frozen=True)
class E2LGModelSettings:
    """Anthropic / generation knobs for the E2LG agent.

    Defaults are tuned for low-latency voice replies: a tight ``max_tokens``
    cap so a single sentence finishes quickly, and a low ``temperature`` so
    the agent stays on-script across turns.
    """

    api_key: str
    model: str = "claude-haiku-4-5"
    max_tokens: int = 160
    temperature: float = 0.3


def render_system_prompt(template: str, scenario: Scenario) -> str:
    """Inject ``scenario`` into the prompt template.

    The template is expected to contain ``{property_info}`` and ``{goals_str}``
    placeholders. Any other ``{...}`` is left untouched (we use
    :py:meth:`str.format_map` with a defaulting mapping rather than
    :py:meth:`str.format` so a stray brace does not raise ``KeyError``).
    """

    class _Defaulting(dict[str, str]):
        def __missing__(self, key: str) -> str:  # noqa: D401 — dict hook
            return "{" + key + "}"

    return template.format_map(
        _Defaulting(
            property_info=scenario.property.render(),
            goals_str=scenario.agent_goals.render(),
        )
    )


def load_prompt_template(path: Path) -> str:
    """Read the system-prompt template from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class E2LGAgent:
    """A scenario-bound E2LG agent ready to be plugged into a pipecat pipeline.

    Build with :meth:`from_scenario`; then read ``agent.llm`` to get the
    pipecat LLM service to wire into your :class:`~pipecat.pipeline.pipeline.Pipeline`.
    The agent itself owns no mutable state — anything turn-scoped (history,
    transcript) lives in the pipecat ``LLMContext`` upstream of ``agent.llm``.
    """

    scenario: Scenario
    settings: E2LGModelSettings
    system_prompt: str
    llm: AnthropicLLMService = field(repr=False, compare=False)

    @classmethod
    def from_scenario(
        cls,
        *,
        scenario: Scenario,
        settings: E2LGModelSettings,
        prompt_template: str,
    ) -> "E2LGAgent":
        """Construct an agent for ``scenario`` using ``prompt_template``."""
        system_prompt = render_system_prompt(prompt_template, scenario)
        logger.info(
            "E2LG agent built scenario={} model={} property_rooms={} goal_rooms={}",
            scenario.scenario_id,
            settings.model,
            len(scenario.property.rooms),
            len(scenario.agent_goals.inform_items),
        )
        llm = AnthropicLLMService(
            api_key=settings.api_key,
            settings=AnthropicLLMService.Settings(
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                system_instruction=system_prompt,
                # Prompt caching keeps the (long, static) system instruction
                # off the per-turn token bill.
                enable_prompt_caching=True,
            ),
        )
        return cls(
            scenario=scenario,
            settings=settings,
            system_prompt=system_prompt,
            llm=llm,
        )

    def opening_directive(self) -> Mapping[str, str]:
        """Return the developer-role message used to seed the first turn.

        Plug into ``LLMContext.add_message`` on ``on_first_participant_joined``
        (or RTVI ``on_client_ready``) so the bot greets the user the moment a
        participant arrives, instead of waiting for them to speak first.
        """
        return {
            "role": "developer",
            "content": (
                "A new visitor has just joined the open house. Greet them in one short sentence "
                "and invite them to begin the tour. Do not describe any room yet."
            ),
        }
