"""Our own representation of a battle state and the actions available in it.

This module has **no** poke-env dependency on purpose. The Showdown adapter
(`showdown.py`) is responsible for translating poke-env's `Battle` object into
these types, and for translating a chosen `Action` back into a Showdown order.

Keeping this seam means: (a) agents are written against a stable, minimal
interface we control, and (b) if we ever replace poke-env (e.g. with
`@pkmn/engine`), only the adapter changes.

For Phase 0 (random play) this representation is intentionally thin. It will
grow as the agent needs more signal (boosts, hazards, move PP, type info, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
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
    # Descriptive metadata (plain values, no poke-env types). Populated for the
    # relevant action type; the rest stay None.
    move_id: Optional[str] = None
    base_power: Optional[float] = None
    species: Optional[str] = None


@dataclass
class ActivePokemon:
    """Minimal view of a Pokémon currently on the field."""

    species: str
    hp_fraction: float
    status: Optional[str] = None


@dataclass
class BattleState:
    """What a policy sees on a given turn."""

    turn: int
    active: Optional[ActivePokemon]
    opponent_active: Optional[ActivePokemon]
    available_actions: list[Action]
    force_switch: bool = False
