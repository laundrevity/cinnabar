"""Tests for the observation encoding (dependency-free; no poke-env / numpy)."""

from cinnabar.encoding import (
    ACTION_DIM,
    GLOBAL_DIM,
    STATUS_ORDER,
    encode_action,
    encode_global,
    featurize,
)
from cinnabar.state import Action, ActionType, ActivePokemon, BattleState


def _move(index, move_id, base_power, move_type, type_multiplier=1.0, category="PHYSICAL", accuracy=1.0):
    return Action(
        index=index,
        type=ActionType.MOVE,
        label=move_id,
        move_id=move_id,
        base_power=base_power,
        move_type=move_type,
        category=category,
        accuracy=accuracy,
        type_multiplier=type_multiplier,
    )


def _state(active=None, opponent=None, actions=None, turn=1, force_switch=False):
    return BattleState(
        turn=turn,
        active=active if active is not None else ActivePokemon("Tauros", 1.0, None, types=("NORMAL",)),
        opponent_active=opponent,
        available_actions=actions or [],
        force_switch=force_switch,
    )


def test_global_vector_length():
    assert len(encode_global(_state())) == GLOBAL_DIM


def test_action_vector_length():
    assert len(encode_action(_move(0, "tackle", 40, "NORMAL"), _state())) == ACTION_DIM


def test_global_encodes_hp_and_force_switch():
    state = _state(
        active=ActivePokemon("Tauros", 0.5, None, types=("NORMAL",)),
        opponent=ActivePokemon("Chansey", 0.25, None, types=("NORMAL",)),
        force_switch=True,
    )
    g = encode_global(state)
    assert g[0] == 0.5  # our hp
    assert g[1] == 0.25  # opp hp
    assert g[-2] == 1.0  # force_switch flag


def test_global_status_one_hot():
    state = _state(active=ActivePokemon("Tauros", 1.0, "PAR", types=("NORMAL",)))
    g = encode_global(state)
    par = STATUS_ORDER.index("PAR")
    # our status one-hot starts after the two hp floats
    assert g[2 + par] == 1.0
    assert g[2 + STATUS_ORDER.index("NONE")] == 0.0


def test_action_base_power_normalised():
    a = encode_action(_move(0, "selfdestruct", 200, "NORMAL"), _state())
    assert a[2] == 1.0  # 200 / 200


def test_action_stab_flag_set_when_types_match():
    state = _state(active=ActivePokemon("Tauros", 1.0, None, types=("NORMAL",)))
    stab = encode_action(_move(0, "bodyslam", 85, "NORMAL"), state)
    no_stab = encode_action(_move(1, "surf", 95, "WATER"), state)
    assert stab[4] == 1.0
    assert no_stab[4] == 0.0


def test_action_status_move_flag():
    a = encode_action(_move(0, "thunderwave", 0, "ELECTRIC", category="STATUS"), _state())
    assert a[5] == 1.0


def test_action_type_multiplier_passthrough():
    a = encode_action(_move(0, "tbolt", 95, "ELECTRIC", type_multiplier=2.0), _state())
    assert a[3] == 2.0


def test_switch_action_features():
    sw = Action(0, ActionType.SWITCH, "switch:Chansey", species="Chansey")
    a = encode_action(sw, _state())
    assert a[0] == 0.0  # not a move
    assert a[1] == 1.0  # is a switch


def test_featurize_shapes():
    actions = [_move(0, "a", 40, "NORMAL"), _move(1, "b", 110, "ICE"),
               Action(2, ActionType.SWITCH, "sw", species="Chansey")]
    g, per_action = featurize(_state(actions=actions))
    assert len(g) == GLOBAL_DIM
    assert len(per_action) == 3
    assert all(len(v) == ACTION_DIM for v in per_action)


def test_encoding_is_deterministic():
    state = _state(actions=[_move(0, "a", 40, "NORMAL")])
    assert featurize(state) == featurize(state)
