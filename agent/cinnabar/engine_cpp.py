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
        mid = _to_id(name)
        m = self._moves.get(mid, {})
        acc = m.get("accuracy", True)
        dmg = m.get("damage")
        fixed = 100.0 if dmg == "level" else (float(dmg) if isinstance(dmg, (int, float)) else None)
        # Effect facts so the policy can distinguish moves with the same power/type (a heal, a
        # sleep-inducer, a paralysis move and a stat-booster otherwise all look like "0-power status").
        smap = {"slp": "SLP", "par": "PAR", "frz": "FRZ", "brn": "BRN", "psn": "PSN", "tox": "PSN"}
        sec = m.get("secondary") or {}
        if m.get("status") in smap:
            eff_status, eff_chance = smap[m["status"]], 1.0
        elif sec.get("status") in smap:
            eff_status, eff_chance = smap[sec["status"]], float(sec.get("chance", 100)) / 100.0
        else:
            eff_status, eff_chance = "", 0.0
        boosts = m.get("boosts") or {}
        sec_boosts = sec.get("boosts") or {}
        target = str(m.get("target", ""))
        return {
            "base_power": float(m.get("basePower", 0) or 0),
            "type": (m.get("type") or "Normal").upper(),
            "category": (m.get("category") or "Status").upper(),
            "accuracy": 1.0 if acc is True else (0.0 if acc is False else float(acc) / 100.0),
            "fixed": fixed,
            "effect_status": eff_status,
            "effect_chance": eff_chance,
            "heals": bool(m.get("heal")) or mid in ("recover", "softboiled", "rest"),
            "boosts_self": bool(boosts) and target == "self",
            "lowers_foe": (bool(boosts) and target != "self" and any(v < 0 for v in boosts.values()))
                          or any(v < 0 for v in sec_boosts.values()),
            "recharge": mid == "hyperbeam" or bool((m.get("flags") or {}).get("recharge"))
                        or (m.get("self") or {}).get("volatileStatus") == "mustrecharge",
            "self_destruct": bool(m.get("selfdestruct")),
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


class Reveal:
    """Per-battle partial-information memory: which opponent mons have been seen, and which moves
    each has been seen using. Accumulates over a battle (monotonic) — the agent's memory."""

    __slots__ = ("mons", "moves")

    def __init__(self) -> None:
        self.mons: set[int] = set()          # revealed opponent team indices
        self.moves: dict[str, set[str]] = {}  # species -> set of revealed move_ids

    def see_move(self, species: str, move_id: str) -> None:
        if move_id and move_id not in ("struggle", "recharge"):
            self.moves.setdefault(species, set()).add(move_id)


def reveal_move(observer: "Reveal | None", mover_state: BattleState, action) -> None:
    """Record (into the observer's memory) that `mover_state`'s active just used `action`."""
    if observer is None or action is None or mover_state.active is None:
        return
    if action.type == ActionType.MOVE and action.move_id:
        observer.see_move(mover_state.active.species, action.move_id)


def build_state(battle, player: int, my_team: Team, static: StaticData, tag: str,
                reveal: "Reveal | None" = None, opp_team: Team | None = None) -> BattleState:
    """Translate the engine's view for `player` into a BattleState.

    Partial information (the real game): when `reveal` — a persistent per-battle set of opponent
    team indices the player has seen — is given, the opponent's bench is restricted to revealed
    mons. The current active is always revealed and the set accumulates across turns, so it
    doubles as the agent's memory of what the opponent has shown so far. `reveal=None` keeps the
    old full-information view.
    """
    ts = battle.team_state(player)        # [(species, hp_frac, status, fainted, active), ...]
    ots = battle.team_state(1 - player)
    if reveal is not None:
        opp_active_idx = next((i for i, e in enumerate(ots) if e[4]), None)
        if opp_active_idx is not None:
            reveal.mons.add(opp_active_idx)
    opp_indices = [i for i in range(len(ots)) if reveal is None or i in reveal.mons]

    def active_view(entries, p: int) -> ActivePokemon | None:
        e = next((x for x in entries if x[4]), None)
        if e is None:
            return None
        conf, refl, lscr, seed, disa, tox, tstage = battle.active_volatiles(p)
        return ActivePokemon(species=e[0], hp_fraction=e[1], status=(e[2] or None),
                             types=static.species_types(e[0]), speed=static.species_speed(e[0]),
                             boosts=tuple(battle.active_boosts(p)),
                             must_recharge=battle.active_must_recharge(p),
                             sleep_turns=battle.active_sleep_turns(p),
                             confused=conf, reflect=refl, light_screen=lscr, leech_seeded=seed,
                             disabled=disa, toxic=tox, tox_stage=tstage)

    opp_entry = next((x for x in ots if x[4]), None)
    opp_types = static.species_types(opp_entry[0]) if opp_entry else ()
    active_pos = next((i for i, e in enumerate(ts) if e[4]), 0)
    my_moves = my_team[active_pos][1]

    actions: list[Action] = []
    for c in battle.choices(player):
        if c.kind == ce.ChoiceKind.Move:
            name = {-1: "Struggle", -2: "Recharge"}.get(c.index) or my_moves[c.index]
            mm = static.move_meta(name)
            actions.append(Action(
                index=len(actions), type=ActionType.MOVE, label=name, move_id=_to_id(name),
                base_power=mm["base_power"], move_type=mm["type"], category=mm["category"],
                accuracy=mm["accuracy"], fixed_damage=mm["fixed"],
                type_multiplier=static.type_mult(mm["type"], opp_types) if opp_types else None,
                effect_status=mm["effect_status"], effect_chance=mm["effect_chance"],
                heals=mm["heals"], boosts_self=mm["boosts_self"], lowers_foe=mm["lowers_foe"],
                recharge=mm["recharge"], self_destruct=mm["self_destruct"],
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
        return TeamMon(species=e[0], hp_fraction=e[1], fainted=e[3], status=(e[2] or None),
                       active=e[4], types=static.species_types(e[0]))

    # Opponent memory: the active opponent's revealed moves as a threat profile (effect features +
    # type-effectiveness vs OUR active). Empty until the opponent has shown this mon use moves.
    opp_revealed: list[Action] = []
    if reveal is not None and opp_team is not None and opp_entry is not None:
        seen = reveal.moves.get(opp_entry[0])
        if seen:
            my_types = static.species_types(ts[active_pos][0])
            spec_moves = next((mvs for sp, mvs in opp_team if sp == opp_entry[0]), [])
            for mv_name in spec_moves:
                if _to_id(mv_name) in seen:
                    mm = static.move_meta(mv_name)
                    opp_revealed.append(Action(
                        index=len(opp_revealed), type=ActionType.MOVE, label=mv_name,
                        move_id=_to_id(mv_name), base_power=mm["base_power"], move_type=mm["type"],
                        category=mm["category"], accuracy=mm["accuracy"], fixed_damage=mm["fixed"],
                        effect_status=mm["effect_status"], recharge=mm["recharge"],
                        type_multiplier=static.type_mult(mm["type"], my_types) if my_types else None,
                    ))

    return BattleState(
        turn=battle.turn, active=active_view(ts, player), opponent_active=active_view(ots, 1 - player),
        available_actions=actions, force_switch=battle.must_switch(player), battle_tag=tag,
        team=[team_mon(e) for e in ts],                          # our own team: fully known
        opponent_team=[team_mon(ots[i]) for i in opp_indices],   # partial info: revealed mons only
        opponent_revealed_moves=opp_revealed,                    # memory: active opp's shown moves
    )


def play_battle(p1_policy, p2_policy, team1: Team, team2: Team, static: StaticData,
                seed: int, tag: str = "eng", turn_limit: int = 1000, clauses: bool = False):
    """Run one battle on the engine, p1_policy vs p2_policy. Returns the ce.Battle so the
    caller can read result() / final_material(). Policies record their own steps (e.g.
    PGPolicy via battle_tag); p1 records under `tag`, p2 under `tag + "_opp"`.
    `clauses` enables OU Sleep + Freeze Clause (match training / the real format)."""
    spec1 = [(s, list(mvs)) for s, mvs in team1]
    spec2 = [(s, list(mvs)) for s, mvs in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)

    r1, r2 = Reveal(), Reveal()  # p1's memory of p2 (and vice-versa) — partial info + revealed moves
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, tag, reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, tag + "_opp", reveal=r2, opp_team=spec1)
        a1 = p1_policy.select_action(s1)
        a2 = p2_policy.select_action(s2)
        reveal_move(r1, s2, a2)  # p1 sees p2's move
        reveal_move(r2, s1, a1)  # p2 sees p1's move
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle


def final_material(battle) -> tuple[float, float]:
    """(our HP-sum, opp HP-sum) over all 6, for end-of-battle reward shaping."""
    p1 = sum(e[1] for e in battle.team_state(0))
    p2 = sum(e[1] for e in battle.team_state(1))
    return p1, p2


def parse_team(text: str) -> Team:
    """Parse a Showdown export-format team into the engine's [(species, [moves]), ...].
    Strips gender/item/ability lines; keeps up to 4 moves per mon."""
    team: Team = []
    for block in text.strip().split("\n\n"):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        species = lines[0].split(" @ ")[0].split(" (")[0].strip()
        moves = [ln[2:].strip() for ln in lines if ln.startswith("- ")]
        if species and moves:
            team.append((species, moves[:4]))
    return team


def load_teams(teams_dir) -> list[Team]:
    """Load every *.txt Showdown team in a directory into engine TeamSpecs."""
    teams = []
    for p in sorted(Path(teams_dir).glob("*.txt")):
        t = parse_team(p.read_text())
        if t:
            teams.append(t)
    return teams
