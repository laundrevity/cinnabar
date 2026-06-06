"""Our own representation of a battle state and the actions available in it.

This module has **no** poke-env dependency on purpose. The Showdown adapter
(`showdown.py`) translates poke-env's `Battle` into these types and a chosen
`Action` back into a Showdown order.

Keeping this seam means agents are written against a stable interface we control,
and swapping the engine later only touches the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionType(Enum):
    MOVE = "move"
    SWITCH = "switch"


@dataclass(frozen=True)
class Action:
    """A single legal choice for the current turn.

    `index` is the action's position in `BattleState.available_actions`; the
    adapter uses it to map this Action back to the underlying Showdown order, so
    the policy never has to touch poke-env objects.
    """

    index: int
    type: ActionType
    label: str
    # Move metadata (None for switches).
    move_id: Optional[str] = None
    base_power: Optional[float] = None
    species: Optional[str] = None
    move_type: Optional[str] = None  # e.g. "ICE", "NORMAL"
    category: Optional[str] = None  # "PHYSICAL" | "SPECIAL" | "STATUS"
    accuracy: Optional[float] = None
    # Type effectiveness of this move vs the current opponent active (a fact from
    # the type chart, like HP). Treat None as 1.0.
    type_multiplier: Optional[float] = None
    fixed_damage: Optional[float] = None  # guaranteed HP for fixed-damage moves (Seismic Toss, Night Shade)
    # Switch metadata (None for moves): what we'd be switching into.
    target_hp_fraction: Optional[float] = None
    target_statused: Optional[bool] = None
    # How hard the opponent active's types hit this switch target (incoming danger;
    # higher = riskier switch). None when there's no opponent to compare to.
    incoming_multiplier: Optional[float] = None


@dataclass
class ActivePokemon:
    """Minimal view of a Pokémon currently on the field."""

    species: str
    hp_fraction: float
    status: Optional[str] = None
    types: tuple[str, ...] = ()  # e.g. ("WATER", "PSYCHIC")
    speed: Optional[int] = None  # base Speed stat (turn order; Gen 1 OU runs maxed stats)


@dataclass
class TeamMon:
    """A team member (ours, or a revealed opponent), for team-state features."""

    species: str
    hp_fraction: float
    fainted: bool
    status: Optional[str] = None
    active: bool = False


@dataclass
class BattleState:
    """What a policy sees on a given turn."""

    turn: int
    active: Optional[ActivePokemon]
    opponent_active: Optional[ActivePokemon]
    available_actions: list[Action]
    force_switch: bool = False
    battle_tag: Optional[str] = None  # groups trajectory steps by battle
    team: list[TeamMon] = field(default_factory=list)  # our full team
    opponent_team: list[TeamMon] = field(default_factory=list)  # opponent's *revealed* mons
