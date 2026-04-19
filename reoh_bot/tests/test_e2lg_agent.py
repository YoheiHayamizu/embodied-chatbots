"""Unit tests for the E2LG agent prompt builder.

Tests focus on pure-functional behaviour: prompt rendering and the
``E2LGAgent.from_scenario`` factory. The Anthropic SDK is constructed but
never called over the network — instantiation alone does not perform any
HTTP request.
"""

from __future__ import annotations

import pytest
from pipecat.services.anthropic.llm import AnthropicLLMService

from reoh_bot.e2lg_agent import (
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
