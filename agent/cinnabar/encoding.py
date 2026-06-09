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

# The 15 Gen 1 types, for per-mon type multi-hots (revealed opponent team).
TYPE_ORDER = ["NORMAL", "FIGHTING", "FLYING", "POISON", "GROUND", "ROCK", "BUG", "GHOST",
              "FIRE", "WATER", "GRASS", "ELECTRIC", "PSYCHIC", "ICE", "DRAGON"]
_TYPE_INDEX = {t: i for i, t in enumerate(TYPE_ORDER)}

TEAM_SIZE = 6
# Per opponent-team slot: present + hp + fainted + statused + a 15-type multi-hot.
_OPP_MON_DIM = 4 + len(TYPE_ORDER)
# Threat profile of the active opponent's revealed moves (memory): count, max power,
# max type-effectiveness vs our active, and has-{sleep, paralysis, any-status, recharge}.
_OPP_MOVE_DIM = 7

# global = hps(2) + two status one-hots + force_switch + turn + team aggregates(4) + speed(1)
#          + active volatiles(12): own+opp boosts(8) + own+opp recharge(2) + own+opp sleep(2)
#          + extra volatiles(12): own+opp {confused, reflect, light_screen, leech_seeded, disabled, toxic}
#          + revealed opponent team: TEAM_SIZE slots x _OPP_MON_DIM (partial-info memory)
#          + active opponent's revealed-move threat profile (_OPP_MOVE_DIM)
#          + clause state(2): do I already have a foe asleep / frozen (my next sleep/freeze fails)
GLOBAL_DIM = (2 + 2 * len(STATUS_ORDER) + 2 + 4 + 1 + 12 + 12
              + TEAM_SIZE * _OPP_MON_DIM + _OPP_MOVE_DIM + 2)
# action = move features(7) + switch-target features(3) + fixed-damage(1) + effect features(11)
#   [status one-hot(5) + chance(1) + heals/boosts_self/lowers_foe/recharge/self_destruct(5)]
#   + will-fail(1): this status move FAILS right now (target already statused, or the
#     Sleep/Freeze Clause is spent) — deterministic Gen 1 mechanics, handed to the net as a fact
ACTION_DIM = 7 + 3 + 1 + 11 + 1

# The status a move can inflict, encoded as a one-hot for the action features.
_EFFECT_STATUS_ORDER = ["SLP", "PAR", "FRZ", "BRN", "PSN"]

_MAX_BASE_POWER = 200.0  # Self-Destruct (200) is about the Gen 1 ceiling
_TURN_SCALE = 50.0


def _effective_speed(mon) -> int | None:
    """Base Speed, quartered if paralyzed (Gen 1 rule). None if unknown."""
    if mon is None or mon.speed is None:
        return None
    return mon.speed // 4 if mon.status == "PAR" else mon.speed


def _speed_advantage(active, opponent) -> float:
    """1.0 if we move first, 0.0 if last, 0.5 on a tie / unknown."""
    a, b = _effective_speed(active), _effective_speed(opponent)
    if a is None or b is None or a == b:
        return 0.5
    return 1.0 if a > b else 0.0


def _status_one_hot(status: str | None) -> list[float]:
    vec = [0.0] * len(STATUS_ORDER)
    key = "NONE" if not status else ("PSN" if status == "TOX" else status)
    vec[_STATUS_INDEX.get(key, 0)] = 1.0
    return vec


def _encode_opp_team(state: BattleState) -> list[float]:
    """Per-mon features for the *revealed* opponent team, padded to TEAM_SIZE slots. Unrevealed
    slots are all-zero, so this vector grows as the battle reveals mons — the agent's memory of
    what the opponent has shown (species type, current HP, status, whether it's fainted)."""
    feats: list[float] = []
    team = state.opponent_team[:TEAM_SIZE]
    for i in range(TEAM_SIZE):
        if i < len(team):
            m = team[i]
            types = [0.0] * len(TYPE_ORDER)
            for t in m.types:
                if t in _TYPE_INDEX:
                    types[_TYPE_INDEX[t]] = 1.0
            feats += [1.0, m.hp_fraction, 1.0 if m.fainted else 0.0, 1.0 if m.status else 0.0, *types]
        else:
            feats += [0.0] * _OPP_MON_DIM
    return feats


def _encode_opp_moves(state: BattleState) -> list[float]:
    """Threat profile of the active opponent's revealed moves (memory). All-zero until the
    opponent has shown this mon use a move; then the agent can recall e.g. 'this thing has
    revealed a strong super-effective hit' or 'it carries Sleep'."""
    rm = state.opponent_revealed_moves
    if not rm:
        return [0.0] * _OPP_MOVE_DIM
    powers = [(a.base_power or 0.0) for a in rm]
    mults = [(a.type_multiplier if a.type_multiplier is not None else 1.0) for a in rm]
    return [
        min(len(rm) / 4.0, 1.0),
        max(powers) / _MAX_BASE_POWER,
        min(max(mults) / 4.0, 1.0),  # how super-effectively it has hit us
        1.0 if any(a.effect_status == "SLP" for a in rm) else 0.0,
        1.0 if any(a.effect_status == "PAR" for a in rm) else 0.0,
        1.0 if any(a.effect_status or a.category == "STATUS" for a in rm) else 0.0,
        1.0 if any(a.recharge for a in rm) else 0.0,
    ]


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

    # Active-mon volatiles: stat stages (setup/drops), recharge (a free turn the foe owes),
    # and remaining sleep turns. Appended after the originals so existing indices are stable.
    def _boosts(mon) -> list[float]:
        b = mon.boosts if (mon and mon.boosts) else (0, 0, 0, 0)
        return [x / 6.0 for x in b]  # stages are -6..+6

    our_recharge = 1.0 if (state.active and state.active.must_recharge) else 0.0
    opp_recharge = 1.0 if (state.opponent_active and state.opponent_active.must_recharge) else 0.0
    our_sleep = (state.active.sleep_turns / 7.0) if state.active else 0.0
    opp_sleep = (state.opponent_active.sleep_turns / 7.0) if state.opponent_active else 0.0

    # Sleep/Freeze Clause state: 1.0 when a foe is already asleep/frozen, so under the OU clauses my
    # next sleep/freeze move FAILS. Without this the net can't tell a sleeping foe from a paralysed
    # one (the opp-team encoding only carries a binary "statused"), so it can't learn to stop
    # spamming sleep into a clause-locked target — the exact bug from the browser loss.
    opp_asleep = 1.0 if any(m.status == "SLP" for m in state.opponent_team) else 0.0
    opp_frozen = 1.0 if any(m.status == "FRZ" for m in state.opponent_team) else 0.0

    # Volatiles the agent was previously blind to: confusion, both screens, Leech Seed, a Disabled
    # slot, and badly-poisoned (toxic, distinct from regular poison). 6 bools per side.
    def _vol(mon) -> list[float]:
        if not mon:
            return [0.0] * 6
        return [float(mon.confused), float(mon.reflect), float(mon.light_screen),
                float(mon.leech_seeded), float(mon.disabled), float(mon.toxic)]

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
        _speed_advantage(state.active, state.opponent_active),
        *_boosts(state.active),
        *_boosts(state.opponent_active),
        our_recharge,
        opp_recharge,
        our_sleep,
        opp_sleep,
        *_encode_opp_team(state),
        *_encode_opp_moves(state),
        # New volatiles appended LAST so older feature indices are stable — lets an old checkpoint
        # warm-start by zero-padding these columns (see agent/pad_checkpoint.py).
        *_vol(state.active),
        *_vol(state.opponent_active),
        opp_asleep,  # clause state (also appended last for stable indices / warm-start)
        opp_frozen,
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

    # Move-effect features: a one-hot for the status it can inflict, the chance, and flags for
    # heal / self-boost / foe-drop / recharge / self-destruct. These let the policy tell apart
    # moves that share power/type (e.g. Recover vs Sleep Powder vs Thunder Wave vs Reflect).
    status_one_hot = [1.0 if action.effect_status == s else 0.0 for s in _EFFECT_STATUS_ORDER]

    # Will-fail: a direct, per-action signal that THIS status move does nothing right now. Two
    # deterministic Gen 1 reasons, same feature slot (it means "this fails", whatever the cause):
    #   (a) the target already has a non-volatile status — a primary status move (TWave, Sleep
    #       Powder, Toxic, ...) always fails on a statused target. Watched failure: TWave clicked
    #       32x into a frozen Chansey / 6x into a paralyzed Eggy, throwing a won endgame.
    #       Damaging moves with secondary status (Body Slam) still deal damage — excluded.
    #   (b) Sleep/Freeze Clause spent: a foe is already asleep/frozen, so a NEW sleep/freeze fails
    #       even on a clean target (the original clause-fail case, 16e5869; 15% re-sleep bug).
    # Same slot as the trained clause-fail feature, so models_cf's learned "will_fail -> bad"
    # weight applies to case (a) immediately; a retrain sharpens it.
    will_fail = 0.0
    if (action.effect_status and action.category == "STATUS" and action.effect_chance >= 0.999
            and state.opponent_active is not None and state.opponent_active.status):
        will_fail = 1.0  # (a) target already statused
    if action.effect_status == "SLP" and any(m.status == "SLP" for m in state.opponent_team):
        will_fail = 1.0  # (b) Sleep Clause spent
    elif action.effect_status == "FRZ" and any(m.status == "FRZ" for m in state.opponent_team):
        will_fail = 1.0  # (b) Freeze Clause spent

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
        (action.fixed_damage or 0.0) / 100.0,  # Seismic Toss/Night Shade = 1.0
        *status_one_hot,
        action.effect_chance,
        1.0 if action.heals else 0.0,
        1.0 if action.boosts_self else 0.0,
        1.0 if action.lowers_foe else 0.0,
        1.0 if action.recharge else 0.0,
        1.0 if action.self_destruct else 0.0,
        will_fail,  # clause-fail (appended last for stable indices / warm-start)
    ]


def featurize(state: BattleState) -> tuple[list[float], list[list[float]]]:
    """Returns (global_vector, [per-action vectors]) for the current turn."""
    return encode_global(state), [encode_action(a, state) for a in state.available_actions]


def stack_global(history: list[list[float]], current: list[float], k: int) -> list[float]:
    """Frame-stack probe: return the k most recent global vectors concatenated (oldest first,
    zero-padded at the front on early turns), then record `current` for next turn. `history`
    holds the previous k-1 globals and is mutated in place. k=1 is a no-op (returns `current`)."""
    if k <= 1:
        return current
    frames = history[-(k - 1):] + [current]
    flat = [0.0] * (len(current) * (k - len(frames)))  # front-pad when fewer than k turns seen
    for fr in frames:
        flat.extend(fr)
    history.append(current)
    del history[:-(k - 1)]  # keep only the previous k-1 for next turn
    return flat
