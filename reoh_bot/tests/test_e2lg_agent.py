"""Unit tests for the E2LG agent prompt builder.

Tests focus on pure-functional behaviour: prompt rendering and the
``E2LGAgent.from_scenario`` factory. The Anthropic SDK is constructed but
never called over the network — instantiation alone does not perform any
HTTP request.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pipecat.services.anthropic.llm import AnthropicLLMService

from reoh_bot.arrival_gate import ArrivalGate
from reoh_bot.e2lg_agent import (
    WAIT_FOR_ARRIVAL_TOOL,
    E2LGAgent,
    E2LGModelSettings,
    render_system_prompt,
)
from reoh_bot.scenarios import AgentGoals, Property, Room, Scenario


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


def test_render_system_prompt_substitutes_placeholders(scenario: Scenario) -> None:
    template = "rooms:\n{property_info}\n\ngoals:\n{goals_str}"
    rendered = render_system_prompt(template, scenario)
    assert "- Entrance: storage=yes, size=medium" in rendered
    assert "- Kitchen: size=large, kitchentype=island" in rendered
    assert "In the kitchen, mention that size is large" in rendered


def test_render_system_prompt_leaves_unknown_braces_intact(scenario: Scenario) -> None:
    """A stray ``{foo}`` in the template must not raise ``KeyError``."""
    rendered = render_system_prompt("hello {who} {property_info}", scenario)
    assert "{who}" in rendered
    assert "Entrance" in rendered


def test_e2lg_agent_factory_builds_anthropic_service(scenario: Scenario) -> None:
    template = "p:\n{property_info}\n\ng:\n{goals_str}"
    settings = E2LGModelSettings(
        api_key="test", model="claude-test", temperature=0.2, max_tokens=100
    )

    agent = E2LGAgent.from_scenario(
        scenario=scenario,
        settings=settings,
        prompt_template=template,
    )

    assert isinstance(agent.llm, AnthropicLLMService)
    inner_settings = agent.llm._settings
    assert inner_settings.model == "claude-test"
    assert inner_settings.temperature == 0.2
    assert inner_settings.max_tokens == 100
    assert inner_settings.enable_prompt_caching is True
    assert "Entrance" in inner_settings.system_instruction

    assert agent.scenario is scenario
    assert agent.settings is settings


def test_opening_directive_is_developer_role(scenario: Scenario) -> None:
    agent = E2LGAgent.from_scenario(
        scenario=scenario,
        settings=E2LGModelSettings(api_key="x"),
        prompt_template="{property_info} {goals_str}",
    )
    msg = agent.opening_directive()
    assert msg["role"] == "developer"
    assert "open house" in msg["content"].lower()


def _build_agent(scenario: Scenario) -> E2LGAgent:
    return E2LGAgent.from_scenario(
        scenario=scenario,
        settings=E2LGModelSettings(api_key="x"),
        prompt_template="{property_info} {goals_str}",
    )


def test_attach_arrival_gate_returns_tool_schema(scenario: Scenario) -> None:
    agent = _build_agent(scenario)
    gate = ArrivalGate()

    tools = agent.attach_arrival_gate(gate)

    assert len(tools.standard_tools) == 1
    schema = tools.standard_tools[0]
    assert schema.name == WAIT_FOR_ARRIVAL_TOOL
    assert schema.required == []
    # The schema must be advertised to the LLM with no required arguments,
    # so the model can call it as a bare ``wait_for_arrival()``.
    assert schema.properties == {}


def test_attach_arrival_gate_registers_handler_on_llm(scenario: Scenario) -> None:
    agent = _build_agent(scenario)
    gate = ArrivalGate()

    agent.attach_arrival_gate(gate)

    # The LLM service stores registered handlers on the private ``_functions``
    # dict; we don't rely on the public surface here because there isn't one
    # for "is this function registered?".
    assert WAIT_FOR_ARRIVAL_TOOL in agent.llm._functions
    entry = agent.llm._functions[WAIT_FOR_ARRIVAL_TOOL]
    # Movement should never be aborted just because the visitor speaks.
    assert entry.cancel_on_interruption is False


def test_arrival_handler_blocks_until_signal(scenario: Scenario) -> None:
    agent = _build_agent(scenario)
    gate = ArrivalGate()
    agent.attach_arrival_gate(gate)

    handler = agent.llm._functions[WAIT_FOR_ARRIVAL_TOOL].handler

    results: list[Any] = []

    async def _result_callback(result: Any, *, properties: Any = None) -> None:  # noqa: ARG001
        results.append(result)

    class _FakeParams:
        def __init__(self) -> None:
            self.function_name = WAIT_FOR_ARRIVAL_TOOL
            self.tool_call_id = "test-1"
            self.arguments: dict[str, Any] = {}
            self.llm = agent.llm
            self.context = None
            self.result_callback = _result_callback

    async def _run() -> None:
        call = asyncio.create_task(handler(_FakeParams()))
        await asyncio.sleep(0)
        assert not call.done(), "handler must block until the gate is signalled"
        assert results == []

        gate.signal()
        await asyncio.wait_for(call, timeout=1.0)
        assert results == [{"status": "arrived"}]

    asyncio.run(_run())
