"""C++ engine adapter — the counterpart to ``showdown.py``, but backed by our own
fast Gen 1 engine (the ``cinnabar_engine`` pybind module) instead of poke-env/Showdown.

It translates the engine's battle state into the agent's ``BattleState``/``Action`` types,
so the engine-free RL core (state / policy / encoding / rl) trains on the fast, fidelity-
validated engine. Battle *dynamics* come from C++ (``team_state``, ``choices``, ``step``);
static move/species/type *data* for featurization comes from poke-env's Gen 1 tables
(cached dict lookups — the speed win is the C++ battle sim, not these).

A team is given as ``[(species, [move_names]), ...]`` (the engine's TeamSpec).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Import the built C++ module from engine/build.
_BUILD = Path(__file__).resolve().parents[2] / "engine" / "build"
if str(_BUILD) not in sys.path:
    sys.path.insert(0, str(_BUILD))
import cinnabar_engine as ce  # noqa: E402

from .state import Action, ActionType, ActivePokemon, BattleState, TeamMon  # noqa: E402

Team = list[tuple[str, list[str]]]


def _to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


class StaticData:
    """Gen 1 move / species / type-chart data from poke-env (static; battles run in C++)."""

    def __init__(self, gen: int = 1) -> None:
        from poke_env.data import GenData

        data = GenData.from_gen(gen)
        self._moves = data.moves or data.load_moves(gen)
        self._dex = data.pokedex or data.load_pokedex(gen)
        raw = data.type_chart or data.load_type_chart(gen)
        # defender-keyed: chart[DEFENDER][ATTACKER] = multiplier (verified in tools/gen_data.py)
        self._chart = {k.upper(): {a.upper(): float(v) for a, v in d.items()} for k, d in raw.items()}

    @lru_cache(maxsize=None)
    def move_meta(self, name: str) -> dict:
        m = self._moves.get(_to_id(name), {})
        acc = m.get("accuracy", True)
        dmg = m.get("damage")
        fixed = 100.0 if dmg == "level" else (float(dmg) if isinstance(dmg, (int, float)) else None)
        return {
            "base_power": float(m.get("basePower", 0) or 0),
            "type": (m.get("type") or "Normal").upper(),
            "category": (m.get("category") or "Status").upper(),
            "accuracy": 1.0 if acc is True else (0.0 if acc is False else float(acc) / 100.0),
            "fixed": fixed,
        }

    @lru_cache(maxsize=None)
    def species_types(self, name: str) -> tuple[str, ...]:
        return tuple(t.upper() for t in self._dex.get(_to_id(name), {}).get("types", []))

    @lru_cache(maxsize=None)
    def species_speed(self, name: str):
        bs = (self._dex.get(_to_id(name), {}).get("baseStats")
              or self._dex.get(_to_id(name), {}).get("base_stats") or {})
        return bs.get("spe")

    def type_mult(self, atk_type: str, def_types: tuple[str, ...]) -> float:
        mult = 1.0
        for dt in def_types:
            mult *= self._chart.get(dt, {}).get(atk_type, 1.0)
        return mult


def build_state(battle, player: int, my_team: Team, static: StaticData, tag: str) -> BattleState:
    """Translate the engine's view for `player` into a BattleState."""
    ts = battle.team_state(player)        # [(species, hp_frac, status, fainted, active), ...]
    ots = battle.team_state(1 - player)

    def active_view(entries) -> ActivePokemon | None:
        e = next((x for x in entries if x[4]), None)
        if e is None:
            return None
        return ActivePokemon(species=e[0], hp_fraction=e[1], status=(e[2] or None),
                             types=static.species_types(e[0]), speed=static.species_speed(e[0]))

    opp_entry = next((x for x in ots if x[4]), None)
    opp_types = static.species_types(opp_entry[0]) if opp_entry else ()
    active_pos = next((i for i, e in enumerate(ts) if e[4]), 0)
    my_moves = my_team[active_pos][1]

    actions: list[Action] = []
    for c in battle.choices(player):
        if c.kind == ce.ChoiceKind.Move:
            name = "Struggle" if c.index < 0 else my_moves[c.index]
            mm = static.move_meta(name)
            actions.append(Action(
                index=len(actions), type=ActionType.MOVE, label=name, move_id=_to_id(name),
                base_power=mm["base_power"], move_type=mm["type"], category=mm["category"],
                accuracy=mm["accuracy"], fixed_damage=mm["fixed"],
                type_multiplier=static.type_mult(mm["type"], opp_types) if opp_types else None,
            ))
        else:  # Switch — c.index is the team position
            tgt = ts[c.index]
            tgt_types = static.species_types(tgt[0])
            incoming = (max((static.type_mult(ot, tgt_types) for ot in opp_types), default=None)
                        if opp_types else None)
            actions.append(Action(
                index=len(actions), type=ActionType.SWITCH, label=f"switch:{tgt[0]}",
                species=tgt[0], target_hp_fraction=tgt[1], target_statused=bool(tgt[2]),
                incoming_multiplier=incoming,
            ))

    def team_mon(e) -> TeamMon:
        return TeamMon(species=e[0], hp_fraction=e[1], fainted=e[3], status=(e[2] or None), active=e[4])

    return BattleState(
        turn=battle.turn, active=active_view(ts), opponent_active=active_view(ots),
        available_actions=actions, force_switch=battle.must_switch(player), battle_tag=tag,
        team=[team_mon(e) for e in ts], opponent_team=[team_mon(e) for e in ots],  # full info (v1)
    )


def play_battle(p1_policy, p2_policy, team1: Team, team2: Team, static: StaticData,
                seed: int, tag: str = "eng", turn_limit: int = 1000):
    """Run one battle on the engine, p1_policy vs p2_policy. Returns the ce.Battle so the
    caller can read result() / final_material(). Policies record their own steps (e.g.
    PGPolicy via battle_tag); p1 records under `tag`, p2 under `tag + "_opp"`."""
    spec1 = [(s, list(mvs)) for s, mvs in team1]
    spec2 = [(s, list(mvs)) for s, mvs in team2]
    battle = ce.make_battle(spec1, spec2, seed)

    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        a1 = p1_policy.select_action(build_state(battle, 0, spec1, static, tag))
        a2 = p2_policy.select_action(build_state(battle, 1, spec2, static, tag + "_opp"))
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle


def final_material(battle) -> tuple[float, float]:
    """(our HP-sum, opp HP-sum) over all 6, for end-of-battle reward shaping."""
    p1 = sum(e[1] for e in battle.team_state(0))
    p2 = sum(e[1] for e in battle.team_state(1))
    return p1, p2
