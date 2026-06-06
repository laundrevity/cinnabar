"""Turn a BattleState into numbers — the observation design for the RL agent.

Dependency-free (plain Python floats, no numpy/torch) so it stays engine-free and
unit-testable anywhere. The agent converts these to tensors.

Two halves, matching the per-action policy architecture:

  * ``encode_global(state)``  -> one fixed-length vector describing the board.
  * ``encode_action(a, s)``   -> one fixed-length vector per *legal* action.

The network scores each action from (global ++ action) features, so we only ever
featurize legal actions and never need a fixed action-index space or a mask.
"""

from __future__ import annotations

from .state import Action, ActionType, BattleState

# Gen 1 statuses (+ a NONE bucket). TOX doesn't exist in Gen 1; fold into PSN.
STATUS_ORDER = ["NONE", "BRN", "FRZ", "PAR", "PSN", "SLP"]
_STATUS_INDEX = {name: i for i, name in enumerate(STATUS_ORDER)}

TEAM_SIZE = 6

# global = hps(2) + two status one-hots + force_switch + turn + team aggregates(4)
GLOBAL_DIM = 2 + 2 * len(STATUS_ORDER) + 2 + 4
# action = move features(7) + switch-target features(3)
ACTION_DIM = 7 + 3

_MAX_BASE_POWER = 200.0  # Self-Destruct (200) is about the Gen 1 ceiling
_TURN_SCALE = 50.0


def _status_one_hot(status: str | None) -> list[float]:
    vec = [0.0] * len(STATUS_ORDER)
    key = "NONE" if not status else ("PSN" if status == "TOX" else status)
    vec[_STATUS_INDEX.get(key, 0)] = 1.0
    return vec


def encode_global(state: BattleState) -> list[float]:
    """Fixed-length board summary (length ``GLOBAL_DIM``)."""
    our_hp = state.active.hp_fraction if state.active else 0.0
    opp_hp = state.opponent_active.hp_fraction if state.opponent_active else 0.0
    our_status = state.active.status if state.active else None
    opp_status = state.opponent_active.status if state.opponent_active else None

    # Team-state aggregates: full info on our side, KO/reveal counts on theirs.
    our_alive = sum(1 for m in state.team if not m.fainted) / TEAM_SIZE
    our_hp_total = sum(m.hp_fraction for m in state.team) / TEAM_SIZE
    opp_fainted = sum(1 for m in state.opponent_team if m.fainted) / TEAM_SIZE
    opp_revealed = len(state.opponent_team) / TEAM_SIZE

    return [
        our_hp,
        opp_hp,
        *_status_one_hot(our_status),
        *_status_one_hot(opp_status),
        1.0 if state.force_switch else 0.0,
        min(state.turn / _TURN_SCALE, 1.0),
        our_alive,
        our_hp_total,
        opp_fainted,
        opp_revealed,
    ]


def _accuracy(value) -> float:
    # poke-env accuracy can be a float, or True for never-miss moves.
    if value is None or value is True:
        return 1.0
    if value is False:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def encode_action(action: Action, state: BattleState) -> list[float]:
    """Fixed-length feature vector for one legal action (length ``ACTION_DIM``)."""
    is_move = action.type == ActionType.MOVE
    base_power = (action.base_power or 0.0) / _MAX_BASE_POWER
    multiplier = action.type_multiplier if action.type_multiplier is not None else 1.0

    stab = 0.0
    if is_move and action.move_type and state.active and action.move_type in state.active.types:
        stab = 1.0

    # Switch-target context (0 for moves): what we'd be switching into and how
    # dangerous the incoming matchup is.
    target_hp = action.target_hp_fraction or 0.0
    target_statused = 1.0 if action.target_statused else 0.0
    incoming = action.incoming_multiplier if action.incoming_multiplier is not None else 0.0

    return [
        1.0 if is_move else 0.0,
        1.0 if action.type == ActionType.SWITCH else 0.0,
        base_power,
        multiplier,
        stab,
        1.0 if action.category == "STATUS" else 0.0,
        _accuracy(action.accuracy),
        target_hp,
        target_statused,
        incoming,
    ]


def featurize(state: BattleState) -> tuple[list[float], list[list[float]]]:
    """Returns (global_vector, [per-action vectors]) for the current turn."""
    return encode_global(state), [encode_action(a, state) for a in state.available_actions]
