"""Cinnabar agent.

The core of the package (``state`` and ``policy``) is deliberately free of any
poke-env / Showdown dependency: agents reason over our own ``BattleState`` and
``Action`` types. The only place that touches poke-env is ``showdown.py`` (the
adapter), so swapping the interface later means rewriting one file, not the agent.
"""

from .state import Action, ActionType, ActivePokemon, BattleState
from .policy import Policy, RandomPolicy

__all__ = [
    "Action",
    "ActionType",
    "ActivePokemon",
    "BattleState",
    "Policy",
    "RandomPolicy",
]
