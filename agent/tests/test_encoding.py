"""Tests for the observation encoding (dependency-free; no poke-env / numpy)."""

from cinnabar.encoding import (
    ACTION_DIM,
    GLOBAL_DIM,
    STATUS_ORDER,
    encode_action,
    encode_global,
    featurize,
)
from cinnabar.state import Action, ActionType, ActivePokemon, BattleState, TeamMon


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
    assert g[14] == 1.0  # force_switch flag (index 14: after 2 hp + 12 status)


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


def test_team_aggregates_in_global():
    team = [
        TeamMon("Tauros", 1.0, fainted=False, active=True),
        TeamMon("Chansey", 0.5, fainted=False),
        TeamMon("Snorlax", 0.0, fainted=True),
    ]
    opp = [
        TeamMon("Tauros", 0.3, fainted=False, active=True),
        TeamMon("Alakazam", 0.0, fainted=True),
    ]
    state = BattleState(turn=1, active=None, opponent_active=None,
                        available_actions=[], team=team, opponent_team=opp)
    g = encode_global(state)
    assert g[16] == 2 / 6    # our mons alive
    assert g[17] == 1.5 / 6  # summed hp 1.0 + 0.5 + 0.0
    assert g[18] == 1 / 6    # opp fainted
    assert g[19] == 2 / 6    # opp revealed


def test_switch_target_features():
    sw = Action(0, ActionType.SWITCH, "switch:Chansey", species="Chansey",
                target_hp_fraction=0.7, target_statused=True, incoming_multiplier=2.0)
    a = encode_action(sw, _state())
    assert a[7] == 0.7  # target hp
    assert a[8] == 1.0  # target statused
    assert a[9] == 2.0  # incoming danger multiplier


def test_move_action_has_zero_switch_features():
    a = encode_action(_move(0, "tackle", 40, "NORMAL"), _state())
    assert a[7] == 0.0 and a[8] == 0.0 and a[9] == 0.0


def test_fixed_damage_feature():
    seismic = Action(0, ActionType.MOVE, "seismictoss", move_id="seismictoss",
                     base_power=0, move_type="FIGHTING", fixed_damage=100.0)
    assert encode_action(seismic, _state())[10] == 1.0  # 100 / 100
    # a normal damaging move has no fixed-damage component
    assert encode_action(_move(0, "psychic", 90, "PSYCHIC"), _state())[10] == 0.0


def test_speed_advantage_feature():
    faster = BattleState(turn=1, available_actions=[],
        active=ActivePokemon("Tauros", 1.0, types=("NORMAL",), speed=110),
        opponent_active=ActivePokemon("Snorlax", 1.0, types=("NORMAL",), speed=30))
    assert encode_global(faster)[20] == 1.0

    slower = BattleState(turn=1, available_actions=[],
        active=ActivePokemon("Snorlax", 1.0, types=("NORMAL",), speed=30),
        opponent_active=ActivePokemon("Tauros", 1.0, types=("NORMAL",), speed=110))
    assert encode_global(slower)[20] == 0.0

    # paralysis quarters our speed (110 // 4 = 27 < 30) -> now slower
    para = BattleState(turn=1, available_actions=[],
        active=ActivePokemon("Tauros", 1.0, status="PAR", types=("NORMAL",), speed=110),
        opponent_active=ActivePokemon("Snorlax", 1.0, types=("NORMAL",), speed=30))
    assert encode_global(para)[20] == 0.0


def _status_move(effect_status, category="STATUS", effect_chance=1.0):
    return Action(0, ActionType.MOVE, "statusmove", move_id="statusmove", base_power=0,
                  move_type="NORMAL", category=category,
                  effect_status=effect_status, effect_chance=effect_chance)


def test_will_fail_on_already_statused_target():
    """A primary status move into an already-statused active is a guaranteed no-op (Gen 1)."""
    par_foe = _state(opponent=ActivePokemon("Exeggutor", 1.0, status="PAR", types=("GRASS", "PSYCHIC")))
    assert encode_action(_status_move("PAR"), par_foe)[-1] == 1.0
    frz_foe = _state(opponent=ActivePokemon("Chansey", 0.9, status="FRZ", types=("NORMAL",)))
    assert encode_action(_status_move("PAR"), frz_foe)[-1] == 1.0
    clean_foe = _state(opponent=ActivePokemon("Exeggutor", 1.0, types=("GRASS", "PSYCHIC")))
    assert encode_action(_status_move("PAR"), clean_foe)[-1] == 0.0


def test_will_fail_not_for_secondary_status_moves():
    """Body Slam still deals damage into a paralyzed target — only pure status moves fail."""
    par_foe = _state(opponent=ActivePokemon("Snorlax", 1.0, status="PAR", types=("NORMAL",)))
    body_slam = Action(0, ActionType.MOVE, "bodyslam", move_id="bodyslam", base_power=85,
                       move_type="NORMAL", category="PHYSICAL",
                       effect_status="PAR", effect_chance=0.3)
    assert encode_action(body_slam, par_foe)[-1] == 0.0


def test_will_fail_on_spent_sleep_clause():
    """A new sleep fails while any foe is already asleep (Sleep Clause), even on a clean target."""
    state = _state(opponent=ActivePokemon("Tauros", 1.0, types=("NORMAL",)))
    state.opponent_team = [TeamMon("Snorlax", 1.0, False, status="SLP"),
                           TeamMon("Tauros", 1.0, False, active=True)]
    assert encode_action(_status_move("SLP"), state)[-1] == 1.0
    # no foe asleep -> sleeping the clean active is fine
    state.opponent_team = [TeamMon("Tauros", 1.0, False, active=True)]
    assert encode_action(_status_move("SLP"), state)[-1] == 0.0
