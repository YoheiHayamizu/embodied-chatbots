"""Scenario loading and rendering.

Reads REOH scenario JSON files (the same format produced by the upstream
``reoh`` project) and renders the property + agent-goal blocks that get
injected into the E2LG system prompt.

Scenario JSONs are not copied into this repo; they are read at runtime from
``Settings.scenario.scenario_dir``. This avoids duplicating the dataset and
keeps the two repositories cleanly separated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Room:
    """A single room and its observable features."""

    name: str
    features: Mapping[str, str]

    def render(self) -> str:
        if not self.features:
            return f"- {self.name.capitalize()}: (no recorded features)"
        feature_str = ", ".join(f"{k}={v}" for k, v in self.features.items())
        return f"- {self.name.capitalize()}: {feature_str}"


@dataclass(frozen=True)
class Property:
    """A property is an ordered collection of rooms."""

    property_id: str
    rooms: tuple[Room, ...]

    def render(self) -> str:
        return "\n".join(room.render() for room in self.rooms)


@dataclass(frozen=True)
class AgentGoals:
    """Facts the agent is required to communicate during the tour."""

    inform_items: Mapping[str, Mapping[str, str]]

    def render(self) -> str:
        if not self.inform_items:
            return "- (no specific facts required — improvise from the property info)"
        lines: list[str] = []
        for room, slots in self.inform_items.items():
            for slot, value in slots.items():
                lines.append(f"- In the {room}, mention that {slot} is {value}.")
        return "\n".join(lines)


@dataclass(frozen=True)
class Scenario:
    """A scenario pairs a property with the agent's tour goals."""

    scenario_id: str
    property: Property
    agent_goals: AgentGoals

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Scenario":
        scenario_id = str(data.get("scenario_id", "unknown"))
        property_data = data.get("property") or {}
        if not isinstance(property_data, Mapping):
            raise ValueError(f"scenario {scenario_id}: 'property' must be an object")
        rooms_raw = property_data.get("rooms") or {}
        if not isinstance(rooms_raw, Mapping):
            raise ValueError(f"scenario {scenario_id}: 'property.rooms' must be an object")

        rooms: list[Room] = []
        for name, features in rooms_raw.items():
            if isinstance(features, Mapping) and "features" in features:
                features = features["features"]  # type: ignore[assignment]
            if not isinstance(features, Mapping):
                raise ValueError(f"scenario {scenario_id}: room {name} has malformed features")
            rooms.append(Room(name=str(name), features={str(k): str(v) for k, v in features.items()}))

        goals_raw = (data.get("goals") or {}).get("agent") or {}  # type: ignore[union-attr]
        if not isinstance(goals_raw, Mapping):
            raise ValueError(f"scenario {scenario_id}: 'goals.agent' must be an object")
        inform = goals_raw.get("inform") or {}
        if not isinstance(inform, Mapping):
            raise ValueError(f"scenario {scenario_id}: 'goals.agent.inform' must be an object")

        normalised_inform: dict[str, dict[str, str]] = {}
        for room, slots in inform.items():
            if not isinstance(slots, Mapping):
                continue
            normalised_inform[str(room)] = {str(k): str(v) for k, v in slots.items()}

        return cls(
            scenario_id=scenario_id,
            property=Property(
                property_id=str(property_data.get("id", "unknown")),
                rooms=tuple(rooms),
            ),
            agent_goals=AgentGoals(inform_items=normalised_inform),
        )


_SCENARIO_ID_PATTERN = re.compile(r"scenario-(\d+)")


def _parse_index(scenario_id: str) -> int | None:
    match = _SCENARIO_ID_PATTERN.search(scenario_id)
    return int(match.group(1)) if match else None


def load_scenario(scenario_dir: Path, scenario_id: str | int | None) -> Scenario:
    """Load a single scenario from ``scenario_dir``.

    Args:
        scenario_dir: Directory containing ``scenario-*.json`` files.
        scenario_id: A full scenario ID (e.g. ``"scenario-0013"``), a short
            string form (``"13"``), or a bare integer (``13``). ``None``
            selects the first scenario by sorted filename — useful for local
            dev.

    Raises:
        FileNotFoundError: If no matching scenario file exists.
    """
    if isinstance(scenario_id, int):
        scenario_id = str(scenario_id)
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    candidates = sorted(scenario_dir.glob("scenario-*.json"))
    if not candidates:
        raise FileNotFoundError(f"No scenario-*.json files in {scenario_dir}")

    if scenario_id is None:
        chosen = candidates[0]
    else:
        # Numeric input ("13") matches by parsed scenario index only.
        # String input ("scenario-0013") matches by parsed index *or* by exact
        # prefix on the filename stem. Substring-on-stem matching is unsafe
        # because the UUID portion of the filename routinely contains digit
        # runs like "13".
        wanted_index = int(scenario_id) if scenario_id.isdigit() else _parse_index(scenario_id)

        chosen = None
        for candidate in candidates:
            stem_index = _parse_index(candidate.stem)
            if wanted_index is not None and stem_index == wanted_index:
                chosen = candidate
                break
            if not scenario_id.isdigit() and candidate.stem.startswith(scenario_id):
                chosen = candidate
                break
        if chosen is None:
            raise FileNotFoundError(
                f"No scenario matching {scenario_id!r} in {scenario_dir}"
            )

    with chosen.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Scenario.from_dict(data)
