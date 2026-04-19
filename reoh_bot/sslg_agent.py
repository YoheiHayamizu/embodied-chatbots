"""Strategy-conditioned Spoken Language Generation (SSLG) agent.

Sister module to :mod:`reoh_bot.e2lg_agent`. Structurally identical — same
scenario-bound system prompt, same Anthropic LLM wrapper, same arrival-gate
tool surface — with one addition: the agent owns a
:class:`reoh_bot.persona_processor.PersonaStrategyProcessor` that must sit
between the user-aggregator and this agent's LLM. The processor
synchronously extracts a persona snapshot from each user utterance and
appends a per-turn ``developer``-role directive to the ``LLMContext`` so
the LLM can adapt its communication strategy.

The SSLG prompt template differs from E2LG's in one section only: it tells
the model to honour the injected directive and never echo the strategy
label. The rest of the tour logic (greeting, room flow, closing) is
reused verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import FunctionCallParams

from reoh_bot.arrival_gate import ArrivalGate
from reoh_bot.e2lg_agent import (
    WAIT_FOR_ARRIVAL_TOOL,
    load_prompt_template,
    render_system_prompt,
)
from reoh_bot.persona import DEFAULT_STRATEGY_WEIGHTS, StrategySelector
from reoh_bot.persona_extractor import (
    PersonaExtractor,
    PersonaExtractorSettings,
    load_extractor_prompt,
)
from reoh_bot.persona_processor import PersonaStrategyProcessor
from reoh_bot.scenarios import Scenario


@dataclass(frozen=True)
class SSLGModelSettings:
    """Anthropic + persona-extractor knobs for the SSLG agent.

    The main-reply model and the persona-extractor model are configured
    independently: the reply model typically benefits from a slightly
    larger context window, while the extractor is cheapest when pinned to
    Haiku with a tight ``max_tokens`` cap.
    """

    api_key: str
    model: str = "claude-haiku-4-5"
    max_tokens: int = 160
    temperature: float = 0.3
    persona_extractor_model: str = "claude-haiku-4-5"
    persona_extractor_max_tokens: int = 256
    persona_extractor_timeout_s: float = 4.0
    persona_min_utterance_tokens: int = 3
    strategy_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STRATEGY_WEIGHTS)
    )
    strategy_selector_seed: int | None = None


@dataclass(frozen=True)
class SSLGAgent:
    """Scenario-bound SSLG agent ready to be plugged into a pipecat pipeline.

    Build with :meth:`from_scenario` passing both the main system-prompt
    template and the persona-extractor prompt. The returned agent exposes
    ``agent.llm`` (insert after ``agent.persona_processor``) and
    ``agent.persona_processor`` (insert after the user aggregator).
    """

    scenario: Scenario
    settings: SSLGModelSettings
    system_prompt: str
    llm: AnthropicLLMService = field(repr=False, compare=False)
    persona_processor: PersonaStrategyProcessor = field(repr=False, compare=False)

    @classmethod
    def from_scenario(
        cls,
        *,
        scenario: Scenario,
        settings: SSLGModelSettings,
        prompt_template: str,
        extractor_prompt: str,
        context: LLMContext,
    ) -> "SSLGAgent":
        """Construct an agent bound to ``scenario`` and ``context``.

        ``context`` is the shared :class:`LLMContext` that the pipeline's
        user-aggregator appends to; the persona processor needs a handle
        to read the latest user message and append the directive.
        """
        system_prompt = render_system_prompt(prompt_template, scenario)
        logger.info(
            "SSLG agent built scenario={} model={} extractor={} property_rooms={} goal_rooms={}",
            scenario.scenario_id,
            settings.model,
            settings.persona_extractor_model,
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
                # Static system prompt stays cached; per-turn directives
                # arrive as developer-role messages in the context rather
                # than by rewriting ``system_instruction``.
                enable_prompt_caching=True,
            ),
        )

        extractor = PersonaExtractor(
            settings=PersonaExtractorSettings(
                api_key=settings.api_key,
                model=settings.persona_extractor_model,
                max_tokens=settings.persona_extractor_max_tokens,
                timeout_s=settings.persona_extractor_timeout_s,
                min_utterance_tokens=settings.persona_min_utterance_tokens,
            ),
            system_prompt=extractor_prompt,
        )

        selector = StrategySelector(
            weights=settings.strategy_weights,
            seed=settings.strategy_selector_seed,
        )

        persona_processor = PersonaStrategyProcessor(
            context=context,
            extractor=extractor,
            selector=selector,
        )

        return cls(
            scenario=scenario,
            settings=settings,
            system_prompt=system_prompt,
            llm=llm,
            persona_processor=persona_processor,
        )

    def attach_arrival_gate(self, gate: ArrivalGate) -> ToolsSchema:
        """Register the ``wait_for_arrival`` tool on this agent's LLM.

        Identical semantics to :meth:`E2LGAgent.attach_arrival_gate`: the
        tool blocks on the operator's Enter press before letting the LLM
        describe the next room. Duplicated here rather than inherited to
        keep each agent a self-contained value object.
        """

        async def _handle(params: FunctionCallParams) -> None:
            logger.info("SSLG: wait_for_arrival invoked; blocking on operator")
            await gate.wait()
            logger.info("SSLG: wait_for_arrival released; resuming tour")
            await params.result_callback({"status": "arrived"})

        self.llm.register_function(
            WAIT_FOR_ARRIVAL_TOOL,
            _handle,
            cancel_on_interruption=False,
        )

        schema = FunctionSchema(
            name=WAIT_FOR_ARRIVAL_TOOL,
            description=(
                "Block until the robot has finished physically moving to the next "
                "room. Call this after telling the visitor 'please follow me' and "
                "BEFORE describing any feature of the next room. Takes no arguments "
                "and returns once the operator confirms arrival."
            ),
            properties={},
            required=[],
        )
        return ToolsSchema(standard_tools=[schema])

    def opening_directive(self) -> Mapping[str, str]:
        """Same opening-turn seed as :class:`E2LGAgent`.

        The persona processor is a no-op on this very first turn (no user
        message in the context yet), so the greeting runs unchanged.
        """
        return {
            "role": "developer",
            "content": (
                "A new visitor has just joined the open house. Greet them in one short sentence "
                "and invite them to begin the tour. Do not describe any room yet."
            ),
        }


__all__ = [
    "SSLGAgent",
    "SSLGModelSettings",
    "load_extractor_prompt",
    "load_prompt_template",
]
