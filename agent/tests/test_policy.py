"""Tests for the engine-independent core (no poke-env / no server needed)."""

import pytest

from cinnabar.policy import MaxDamagePolicy, RandomPolicy
from cinnabar.state import Action, ActionType, ActivePokemon, BattleState


def _move(index, move_id, base_power, move_type, type_multiplier=1.0):
    return Action(
        index=index,
        type=ActionType.MOVE,
        label=move_id,
        move_id=move_id,
        base_power=base_power,
        move_type=move_type,
        type_multiplier=type_multiplier,
    )


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


# --- MaxDamagePolicy --------------------------------------------------------


def test_maxdamage_picks_higher_base_power():
    active = ActivePokemon("Tauros", 1.0, types=("NORMAL",))
    actions = [
        _move(0, "tackle", 40, "FIGHTING"),  # 40 * 1 * 1   = 40
        _move(1, "blizzard", 110, "ICE"),    # 110 * 1 * 1  = 110
    ]
    state = BattleState(turn=1, active=active, opponent_active=None, available_actions=actions)
    assert MaxDamagePolicy().select_action(state).move_id == "blizzard"


def test_maxdamage_applies_stab():
    active = ActivePokemon("Tauros", 1.0, types=("NORMAL",))
    actions = [
        _move(0, "bodyslam", 85, "NORMAL"),  # 85 * 1 * 1.5 = 127.5  (STAB)
        _move(1, "surf", 95, "WATER"),       # 95 * 1 * 1   = 95
    ]
    state = BattleState(turn=1, active=active, opponent_active=None, available_actions=actions)
    assert MaxDamagePolicy().select_action(state).move_id == "bodyslam"


def test_maxdamage_respects_type_multiplier():
    active = ActivePokemon("Starmie", 1.0, types=("WATER", "PSYCHIC"))
    actions = [
        _move(0, "thunderbolt", 95, "ELECTRIC", type_multiplier=2.0),  # 95 * 2 * 1   = 190
        _move(1, "surf", 95, "WATER", type_multiplier=1.0),            # 95 * 1 * 1.5 = 142.5 (STAB)
    ]
    state = BattleState(turn=1, active=active, opponent_active=None, available_actions=actions)
    assert MaxDamagePolicy().select_action(state).move_id == "thunderbolt"


def test_maxdamage_avoids_immune_moves():
    active = ActivePokemon("Gengar", 1.0, types=("GHOST", "POISON"))
    actions = [
        _move(0, "earthquake", 100, "GROUND", type_multiplier=0.0),  # 100 * 0 = 0 (immune)
        _move(1, "nightshade", 1, "GHOST", type_multiplier=1.0),     # 1 * 1 * 1.5 = 1.5
    ]
    state = BattleState(turn=1, active=active, opponent_active=None, available_actions=actions)
    assert MaxDamagePolicy().select_action(state).move_id == "nightshade"


def test_maxdamage_switches_when_no_damaging_move():
    active = ActivePokemon("Tauros", 1.0, types=("NORMAL",))
    actions = [Action(0, ActionType.SWITCH, "switch:Chansey", species="Chansey")]
    state = BattleState(turn=1, active=active, opponent_active=None, available_actions=actions)
    assert MaxDamagePolicy().select_action(state).type == ActionType.SWITCH


def test_maxdamage_raises_without_actions():
    empty = BattleState(turn=1, active=None, opponent_active=None, available_actions=[])
    with pytest.raises(ValueError):
        MaxDamagePolicy().select_action(empty)
