"""Unit tests for :class:`reoh_bot.sslg_agent.SSLGAgent`.

Mirrors ``test_e2lg_agent.py``: no network calls are made — constructing
the Anthropic service only allocates a client object — and the persona
processor is validated by injecting a shared ``LLMContext`` and inspecting
its state after the factory call.
"""

from __future__ import annotations

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService

from reoh_bot.arrival_gate import ArrivalGate
from reoh_bot.e2lg_agent import WAIT_FOR_ARRIVAL_TOOL
from reoh_bot.persona_processor import PersonaStrategyProcessor
from reoh_bot.scenarios import AgentGoals, Property, Room, Scenario
from reoh_bot.sslg_agent import SSLGAgent, SSLGModelSettings


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        scenario_id="scenario-test",
        property=Property(
            property_id="home-test",
            rooms=(
                Room(name="entrance", features={"storage": "yes", "size": "medium"}),
                Room(name="kitchen", features={"size": "large", "kitchentype": "island"}),
            ),
        ),
        agent_goals=AgentGoals(inform_items={"kitchen": {"size": "large"}}),
    )


def _build_agent(scenario: Scenario, *, seed: int | None = 0) -> tuple[SSLGAgent, LLMContext]:
    template = "p:\n{property_info}\n\ng:\n{goals_str}"
    extractor_prompt = "persona extractor prompt"
    context = LLMContext()
    agent = SSLGAgent.from_scenario(
        scenario=scenario,
        settings=SSLGModelSettings(api_key="test", strategy_selector_seed=seed),
        prompt_template=template,
        extractor_prompt=extractor_prompt,
        context=context,
    )
    return agent, context


def test_sslg_agent_factory_builds_anthropic_service(scenario: Scenario) -> None:
    agent, _ = _build_agent(scenario)

    assert isinstance(agent.llm, AnthropicLLMService)
    inner = agent.llm._settings
    assert inner.enable_prompt_caching is True
    assert "Entrance" in inner.system_instruction
    assert "Kitchen" in inner.system_instruction


def test_sslg_agent_renders_strategy_section_in_prompt(scenario: Scenario) -> None:
    # The SSLG system prompt must advise the model about the per-turn
    # directive; otherwise the strategy behaviour silently degrades.
    from reoh_bot.config import DEFAULT_SSLG_PROMPT_PATH
    from reoh_bot.sslg_agent import load_prompt_template

    template = load_prompt_template(DEFAULT_SSLG_PROMPT_PATH)
    assert "developer" in template.lower()
    assert "approach" in template.lower()


def test_sslg_agent_wires_persona_processor_to_context(scenario: Scenario) -> None:
    agent, context = _build_agent(scenario)

    assert isinstance(agent.persona_processor, PersonaStrategyProcessor)
    assert agent.persona_processor.context is context
    assert agent.persona_processor.extractor is not None
    assert agent.persona_processor.selector is not None


def test_sslg_agent_opening_directive_is_developer_role(scenario: Scenario) -> None:
    agent, _ = _build_agent(scenario)
    msg = agent.opening_directive()
    assert msg["role"] == "developer"
    assert "open house" in msg["content"].lower()


def test_sslg_agent_attach_arrival_gate_registers_tool(scenario: Scenario) -> None:
    agent, _ = _build_agent(scenario)
    gate = ArrivalGate()

    tools = agent.attach_arrival_gate(gate)

    assert len(tools.standard_tools) == 1
    assert tools.standard_tools[0].name == WAIT_FOR_ARRIVAL_TOOL
    assert WAIT_FOR_ARRIVAL_TOOL in agent.llm._functions


def test_sslg_agent_preserves_prompt_caching(scenario: Scenario) -> None:
    # Regression guard: the whole point of keeping the static system
    # prompt in ``system_instruction`` is that Anthropic caches it. If a
    # future change accidentally disables caching, the per-turn directive
    # design becomes a cost regression.
    agent, _ = _build_agent(scenario)
    assert agent.llm._settings.enable_prompt_caching is True


def test_sslg_agent_strategy_selector_respects_seed(scenario: Scenario) -> None:
    a, _ = _build_agent(scenario, seed=42)
    b, _ = _build_agent(scenario, seed=42)

    # Same seed, same weights -> same first choice.
    from reoh_bot.persona import EMPTY_PERSONA

    assert a.persona_processor.selector.select(
        EMPTY_PERSONA, dialog_act="inform"
    ) == b.persona_processor.selector.select(EMPTY_PERSONA, dialog_act="inform")
