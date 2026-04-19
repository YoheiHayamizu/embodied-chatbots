"""Unit tests for scenario loading and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reoh_bot.scenarios import (
    AgentGoals,
    Property,
    Room,
    Scenario,
    load_scenario,
)


def _write_scenario(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def sample_payload() -> dict:
    return {
        "scenario_id": "scenario-0042",
        "property": {
            "id": "home-test-01",
            "rooms": {
                "entrance": {"storage": "yes", "size": "medium"},
                "kitchen": {"kitchentype": "island", "size": "large"},
            },
        },
        "goals": {
            "agent": {
                "inform": {
                    "entrance": {"size": "medium"},
                    "kitchen": {"kitchentype": "island"},
                }
            }
        },
    }


def test_scenario_from_dict_round_trip(sample_payload: dict) -> None:
    scenario = Scenario.from_dict(sample_payload)
    assert scenario.scenario_id == "scenario-0042"
    assert scenario.property.property_id == "home-test-01"
    assert [r.name for r in scenario.property.rooms] == ["entrance", "kitchen"]
    assert scenario.agent_goals.inform_items["kitchen"] == {"kitchentype": "island"}


def test_property_render_uses_capitalised_names() -> None:
    prop = Property(
        property_id="x",
        rooms=(
            Room(name="entrance", features={"storage": "yes"}),
            Room(name="kitchen", features={"size": "large", "kitchentype": "island"}),
        ),
    )
    rendered = prop.render()
    assert "- Entrance: storage=yes" in rendered
    assert "- Kitchen: size=large, kitchentype=island" in rendered


def test_agent_goals_render_to_natural_language() -> None:
    goals = AgentGoals(inform_items={"livingroom": {"view": "yard"}})
    rendered = goals.render()
    assert rendered == "- In the livingroom, mention that view is yard."


def test_agent_goals_render_when_empty() -> None:
    rendered = AgentGoals(inform_items={}).render()
    assert "no specific facts required" in rendered


def test_load_scenario_by_full_id(tmp_path: Path, sample_payload: dict) -> None:
    _write_scenario(tmp_path, "scenario-0042-test.json", sample_payload)
    loaded = load_scenario(tmp_path, "scenario-0042")
    assert loaded.scenario_id == "scenario-0042"


def test_load_scenario_by_short_index(tmp_path: Path, sample_payload: dict) -> None:
    _write_scenario(tmp_path, "scenario-0042-test.json", sample_payload)
    loaded = load_scenario(tmp_path, "42")
    assert loaded.scenario_id == "scenario-0042"


def test_load_scenario_default_picks_first(tmp_path: Path, sample_payload: dict) -> None:
    second = dict(sample_payload, scenario_id="scenario-0001")
    _write_scenario(tmp_path, "scenario-0001-test.json", second)
    _write_scenario(tmp_path, "scenario-0042-test.json", sample_payload)
    loaded = load_scenario(tmp_path, None)
    assert loaded.scenario_id == "scenario-0001"


def test_load_scenario_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario(tmp_path / "nope", None)


def test_load_scenario_no_match(tmp_path: Path, sample_payload: dict) -> None:
    _write_scenario(tmp_path, "scenario-0042-test.json", sample_payload)
    with pytest.raises(FileNotFoundError):
        load_scenario(tmp_path, "999")


def test_scenario_rejects_non_object_property() -> None:
    with pytest.raises(ValueError):
        Scenario.from_dict({"scenario_id": "x", "property": "oops"})


def test_load_scenario_short_index_does_not_substring_match_uuid(tmp_path: Path, sample_payload: dict) -> None:
    """Numeric input must not match digits inside the UUID portion of a filename.

    Regression test: a filename like ``scenario-0003-...-6bc013fa-....json``
    must not be returned for ``scenario_id="13"`` just because "13" appears
    inside the UUID.
    """
    decoy = dict(sample_payload, scenario_id="scenario-0003")
    correct = dict(sample_payload, scenario_id="scenario-0013")
    _write_scenario(tmp_path, "scenario-0003-home-6bc013fa-aaaa.json", decoy)
    _write_scenario(tmp_path, "scenario-0013-home-deadbeef-bbbb.json", correct)
    loaded = load_scenario(tmp_path, "13")
    assert loaded.scenario_id == "scenario-0013"


def test_load_scenario_accepts_int(tmp_path: Path, sample_payload: dict) -> None:
    _write_scenario(tmp_path, "scenario-0042-test.json", sample_payload)
    loaded = load_scenario(tmp_path, 42)
    assert loaded.scenario_id == "scenario-0042"


def test_scenario_handles_legacy_room_features_envelope() -> None:
    """Older scenario JSONs wrap features under a `features` key."""
    payload = {
        "scenario_id": "scenario-9999",
        "property": {
            "id": "legacy",
            "rooms": {
                "entrance": {"room_name": "entrance", "features": {"size": "small"}},
            },
        },
        "goals": {"agent": {"inform": {}}},
    }
    scenario = Scenario.from_dict(payload)
    assert scenario.property.rooms[0].features == {"size": "small"}
