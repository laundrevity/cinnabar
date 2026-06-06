"""Policies: the decision-making core.

A `Policy` maps a `BattleState` to an `Action`. This is the seam every future
agent plugs into — heuristic baselines (Phase 1), RL agents (Phase 2), self-play
(Phase 3) — all implement this one method. No poke-env here.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from .state import Action, ActionType, BattleState


class Policy(ABC):
    """Maps a battle state to a chosen action."""

    @abstractmethod
    def select_action(self, state: BattleState) -> Action:
        ...


class RandomPolicy(Policy):
    """Picks uniformly at random among the legal actions. The Phase 0 baseline."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: BattleState) -> Action:
        if not state.available_actions:
            raise ValueError("RandomPolicy called with no available actions")
        return self._rng.choice(state.available_actions)


class MaxDamagePolicy(Policy):
    """Type-aware max-damage baseline (Phase 1).

    Scores each available move as::

        estimate = base_power * type_multiplier * STAB

    where STAB is 1.5 when the move's type matches one of the active Pokémon's
    types, and plays the highest-scoring move. Status / ineffective moves score 0
    and are avoided. On a turn with no damaging move (e.g. a forced switch) it
    switches; if it can't, it falls back to any legal action.

    Deliberately simple: no switch evaluation, no opponent prediction, no PP or
    accuracy weighting yet. Its only job is to be a fixed, clearly-better-than-
    random yardstick for everything that comes after.
    """

    STAB_MULTIPLIER = 1.5

    def select_action(self, state: BattleState) -> Action:
        if not state.available_actions:
            raise ValueError("MaxDamagePolicy called with no available actions")

        moves = [a for a in state.available_actions if a.type == ActionType.MOVE]
        if moves:
            best = max(moves, key=lambda a: self._estimate(a, state))
            if self._estimate(best, state) > 0:
                return best

        switches = [a for a in state.available_actions if a.type == ActionType.SWITCH]
        if switches:
            return switches[0]

        return state.available_actions[0]

    def _estimate(self, action: Action, state: BattleState) -> float:
        base = action.base_power or 0.0
        multiplier = action.type_multiplier if action.type_multiplier is not None else 1.0
        stab = 1.0
        active = state.active
        if active and action.move_type and action.move_type in active.types:
            stab = self.STAB_MULTIPLIER
        return base * multiplier * stab
