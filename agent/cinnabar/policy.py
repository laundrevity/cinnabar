"""Policies: the decision-making core.

A `Policy` maps a `BattleState` to an `Action`. This is the seam every future
agent plugs into — heuristic baselines (Phase 1), RL agents (Phase 2), self-play
(Phase 3) — all implement this one method. No poke-env here.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from .state import Action, BattleState


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
