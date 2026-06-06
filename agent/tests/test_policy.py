"""Tests for the engine-independent core (no poke-env / no server needed)."""

import pytest

from cinnabar.policy import RandomPolicy
from cinnabar.state import Action, ActionType, ActivePokemon, BattleState


def _sample_state() -> BattleState:
    actions = [
        Action(0, ActionType.MOVE, "bodyslam", move_id="bodyslam", base_power=85),
        Action(1, ActionType.MOVE, "blizzard", move_id="blizzard", base_power=110),
        Action(2, ActionType.SWITCH, "switch:Chansey", species="Chansey"),
    ]
    return BattleState(
        turn=3,
        active=ActivePokemon("Tauros", 1.0, None),
        opponent_active=ActivePokemon("Snorlax", 0.5, "PAR"),
        available_actions=actions,
    )


def test_random_policy_only_picks_legal_actions():
    state = _sample_state()
    policy = RandomPolicy(seed=0)
    chosen = {policy.select_action(state).index for _ in range(1000)}
    assert chosen == {0, 1, 2}


def test_random_policy_is_seeded():
    state = _sample_state()
    a, b = RandomPolicy(seed=42), RandomPolicy(seed=42)
    assert [a.select_action(state).label for _ in range(20)] == [
        b.select_action(state).label for _ in range(20)
    ]


def test_random_policy_raises_without_actions():
    empty = BattleState(turn=1, active=None, opponent_active=None, available_actions=[])
    with pytest.raises(ValueError):
        RandomPolicy().select_action(empty)
