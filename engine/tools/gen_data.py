#!/usr/bin/env python3
"""Generate the engine's Gen 1 static data from the oracle (Showdown, via poke-env),
and audit our hand-coded type chart against it.

Run from the agent env (which has poke-env), e.g. from `agent/`:

    uv run python ../engine/tools/gen_data.py

It:
  1. Audits our (former) hand-coded type chart against poke-env's.
  2. Writes engine/include/cinnabar/gen1_data.hpp with the type chart, the 151 base
     species' stats, and the full Gen 1 move table (power/accuracy/type + the effects
     the engine models: status, stat-stage boosts, heal/rest, fixed damage, self-destruct,
     reflect, high crit-ratio).

"generate data from Showdown, don't hand-type it" — the engine's fidelity rule.
"""

from __future__ import annotations

from pathlib import Path

from poke_env.data import GenData

ORDER = ["NORMAL", "FIGHTING", "FLYING", "POISON", "GROUND", "ROCK", "BUG", "GHOST",
         "FIRE", "WATER", "GRASS", "ELECTRIC", "PSYCHIC", "ICE", "DRAGON"]
ENUM = {t: "Type::" + t.capitalize() for t in ORDER}

OURS_ENTRIES = [
    ("NORMAL", "ROCK", 50), ("NORMAL", "GHOST", 0),
    ("FIGHTING", "NORMAL", 200), ("FIGHTING", "ROCK", 200), ("FIGHTING", "ICE", 200),
    ("FIGHTING", "FLYING", 50), ("FIGHTING", "POISON", 50), ("FIGHTING", "BUG", 50),
    ("FIGHTING", "PSYCHIC", 50), ("FIGHTING", "GHOST", 0),
    ("FLYING", "FIGHTING", 200), ("FLYING", "BUG", 200), ("FLYING", "GRASS", 200),
    ("FLYING", "ROCK", 50), ("FLYING", "ELECTRIC", 50),
    ("POISON", "BUG", 200), ("POISON", "GRASS", 200), ("POISON", "POISON", 50),
    ("POISON", "GROUND", 50), ("POISON", "ROCK", 50), ("POISON", "GHOST", 50),
    ("GROUND", "POISON", 200), ("GROUND", "ROCK", 200), ("GROUND", "FIRE", 200),
    ("GROUND", "ELECTRIC", 200), ("GROUND", "GRASS", 50), ("GROUND", "BUG", 50),
    ("GROUND", "FLYING", 0),
    ("ROCK", "FLYING", 200), ("ROCK", "BUG", 200), ("ROCK", "FIRE", 200),
    ("ROCK", "ICE", 200), ("ROCK", "FIGHTING", 50), ("ROCK", "GROUND", 50),
    ("BUG", "POISON", 200), ("BUG", "GRASS", 200), ("BUG", "PSYCHIC", 200),
    ("BUG", "FIGHTING", 50), ("BUG", "FLYING", 50), ("BUG", "GHOST", 50), ("BUG", "FIRE", 50),
    ("GHOST", "GHOST", 200), ("GHOST", "NORMAL", 0), ("GHOST", "PSYCHIC", 0),
    ("FIRE", "BUG", 200), ("FIRE", "GRASS", 200), ("FIRE", "ICE", 200),
    ("FIRE", "ROCK", 50), ("FIRE", "FIRE", 50), ("FIRE", "WATER", 50), ("FIRE", "DRAGON", 50),
    ("WATER", "GROUND", 200), ("WATER", "ROCK", 200), ("WATER", "FIRE", 200),
    ("WATER", "WATER", 50), ("WATER", "GRASS", 50), ("WATER", "DRAGON", 50),
    ("GRASS", "GROUND", 200), ("GRASS", "ROCK", 200), ("GRASS", "WATER", 200),
    ("GRASS", "FLYING", 50), ("GRASS", "POISON", 50), ("GRASS", "BUG", 50),
    ("GRASS", "FIRE", 50), ("GRASS", "GRASS", 50), ("GRASS", "DRAGON", 50),
    ("ELECTRIC", "FLYING", 200), ("ELECTRIC", "WATER", 200), ("ELECTRIC", "GRASS", 50),
    ("ELECTRIC", "ELECTRIC", 50), ("ELECTRIC", "DRAGON", 50), ("ELECTRIC", "GROUND", 0),
    ("PSYCHIC", "FIGHTING", 200), ("PSYCHIC", "POISON", 200), ("PSYCHIC", "PSYCHIC", 50),
    ("ICE", "FLYING", 200), ("ICE", "GROUND", 200), ("ICE", "GRASS", 200),
    ("ICE", "DRAGON", 200), ("ICE", "WATER", 50), ("ICE", "ICE", 50),
    ("DRAGON", "DRAGON", 200),
]

# Showdown status id -> engine Effect (toxic collapses to plain poison; counter not modelled).
STATUS_EFFECT = {"par": "Effect::Paralyze", "slp": "Effect::Sleep", "frz": "Effect::Freeze",
                 "brn": "Effect::Burn", "psn": "Effect::Poison", "tox": "Effect::Poison"}
# Boost stat key -> engine boost_stat index (Gen 1 special spa/spd both map to one "spc").
BOOST_IDX = {"atk": 0, "def": 1, "spa": 2, "spd": 2, "spc": 2, "spe": 3, "accuracy": 4, "evasion": 5}
CAT_ENUM = {"physical": "Category::Physical", "special": "Category::Special", "status": "Category::Status"}

# Moves that do NOT reset the battle's lastDamage (Gen 1 scripts.ts SKIP_LASTDAMAGE): Counter reads
# it, the rest are status moves. Note e.g. Sleep Powder is NOT here, so it DOES reset lastDamage.
SKIP_LASTDAMAGE = {
    "confuseray", "conversion", "counter", "focusenergy", "glare", "haze", "leechseed",
    "lightscreen", "mimic", "mist", "poisongas", "poisonpowder", "recover", "reflect", "rest",
    "softboiled", "splash", "stunspore", "substitute", "supersonic", "teleport", "thunderwave",
    "toxic", "transform",
}


def ours_matrix() -> dict[tuple[str, str], int]:
    m = {(a, d): 100 for a in ORDER for d in ORDER}
    for a, d, v in OURS_ENTRIES:
        m[(a, d)] = v
    return m


def correct_matrix(gen1: GenData) -> dict[tuple[str, str], int]:
    raw = gen1.type_chart or gen1.load_type_chart(1)
    chart = {k.upper(): {a.upper(): float(v) for a, v in d.items()} for k, d in raw.items()}

    def defender_outer(atk, dfn):
        return chart.get(dfn, {}).get(atk, 1.0)

    if abs(defender_outer("WATER", "FIRE") - 2.0) > 1e-6:
        raise SystemExit("Unexpected type_chart orientation; inspect gen1.type_chart manually.")
    return {(a, d): round(defender_outer(a, d) * 100) for a in ORDER for d in ORDER}


def type_to_enum(t: str) -> str:
    return ENUM.get((t or "").upper(), "Type::None")


def pick_boost(boosts: dict) -> tuple[int, int]:
    """Representative (stat_index, stages) for a boosts dict, or (-1, 0)."""
    for k in ("atk", "def", "spa", "spd", "spc", "spe", "accuracy", "evasion"):
        if k in boosts and boosts[k]:
            return BOOST_IDX[k], int(boosts[k])
    return -1, 0


def translate_move(mid: str, m: dict) -> tuple:
    name = (m.get("name") or mid).replace('"', '\\"')
    typ = type_to_enum(m.get("type", "Normal"))
    cat = CAT_ENUM.get(str(m.get("category", "Status")).lower(), "Category::Status")
    power = int(m.get("basePower", m.get("base_power", 0)) or 0)
    acc = m.get("accuracy", True)
    accuracy = 0 if acc is True else int(acc)  # 0 means "always hits", no accuracy roll
    fixed = 0
    dmg = m.get("damage")
    if dmg == "level":
        fixed = 100  # engine is L100 (Seismic Toss, Night Shade)
    elif isinstance(dmg, int):
        fixed = dmg  # Dragon Rage 40, Sonic Boom 20
    effect, effect_chance = "Effect::None", 0
    bstat, bstages, bfoe, bchance = -1, 0, "false", 0
    high_crit = "true" if int(m.get("critRatio", 1) or 1) >= 2 else "false"
    target = str(m.get("target", ""))

    if mid == "rest":
        effect, effect_chance = "Effect::Rest", 100
    elif m.get("selfdestruct"):
        effect = "Effect::SelfDestruct"
    elif mid in ("recover", "softboiled") or m.get("heal"):
        effect, effect_chance = "Effect::Heal", 100  # 50% heal (poke-env omits Recover's heal field)
    elif mid == "reflect":
        effect, effect_chance = "Effect::Reflect", 100
    elif mid == "substitute":
        effect, effect_chance = "Effect::Substitute", 100
    elif m.get("volatileStatus") == "confusion":
        effect, effect_chance = "Effect::Confuse", 100  # Confuse Ray
    elif mid == "counter":
        effect, effect_chance = "Effect::Counter", 100
    elif m.get("volatileStatus") == "partiallytrapped":
        effect, effect_chance = "Effect::Trap", 100  # Wrap / Bind / Fire Spin / Clamp
    elif m.get("status") in STATUS_EFFECT:
        effect, effect_chance = STATUS_EFFECT[m["status"]], 100
    elif m.get("boosts"):
        idx, st = pick_boost(m["boosts"])
        if idx >= 0:
            bstat, bstages = idx, st
            bfoe = "false" if target == "self" else "true"  # Growl/Screech lower the foe
            bchance = 0  # primary boost: accuracy-gated, no separate boost roll
    else:
        sec = m.get("secondary") or {}
        if sec.get("status") in STATUS_EFFECT:
            effect, effect_chance = STATUS_EFFECT[sec["status"]], int(sec.get("chance", 100))
        elif sec.get("boosts"):
            idx, st = pick_boost(sec["boosts"])
            if idx >= 0:
                bstat, bstages, bfoe, bchance = idx, st, "true", int(sec.get("chance", 100))

    pp = int(m.get("pp", 0) or 0)
    # Recharge (Hyper Beam): prefer Showdown's data (flags.recharge / self.volatileStatus), but
    # fall back to the move id — Hyper Beam is the only Gen 1 recharge move, and poke-env may not
    # surface `flags`/`self`.
    flags = m.get("flags") or {}
    selfvol = (m.get("self") or {}).get("volatileStatus")
    recharge = "true" if (flags.get("recharge") or selfvol == "mustrecharge"
                          or mid == "hyperbeam") else "false"
    rec = m.get("recoil")  # [num, den] e.g. Double-Edge [33,100], Take Down [1,4], Struggle [1,2]
    recoil_num, recoil_den = ((int(rec[0]), int(rec[1]))
                              if isinstance(rec, (list, tuple)) and len(rec) == 2 else (0, 0))
    # ignoreImmunity: explicit flag if set, else status moves default to true (Showdown runImmunity).
    # So Thunder Wave (explicit false) respects type immunity; Confuse Ray / Glare ignore it.
    ig = m.get("ignoreImmunity")
    ignore_imm = ("true" if cat == "Category::Status" else "false") if ig is None else ("true" if ig else "false")
    priority = int(m.get("priority", 0) or 0)            # Counter = -5 (moves last)
    skip_ld = "true" if mid in SKIP_LASTDAMAGE else "false"  # does not reset battle.last_damage
    return (name, typ, cat, power, accuracy, fixed, effect, effect_chance, bstat, bstages, bfoe,
            bchance, high_crit, pp, recharge, recoil_num, recoil_den, ignore_imm, priority, skip_ld)


def build_lines(correct, dex, moves) -> tuple[list[str], int, int]:
    lines = [
        "// AUTO-GENERATED by engine/tools/gen_data.py from Showdown (poke-env) Gen 1 data.",
        "// Do not edit by hand; regenerate instead.",
        "#pragma once",
        '#include "cinnabar/engine.hpp"',
        "",
        "namespace cinnabar {",
        "",
        "// Type effectiveness x100, indexed [attacking][defending] in Type enum order.",
        "inline const int GEN1_TYPE_CHART[15][15] = {",
    ]
    for a in ORDER:
        row = ", ".join(f"{correct[(a, d)]:3d}" for d in ORDER)
        lines.append(f"    {{{row}}},  // {a}")
    lines += ["};", ""]

    lines.append("struct SpeciesEntry { const char* name; Type t1, t2; int hp, atk, def, spc, spe; };")
    lines.append("inline const SpeciesEntry GEN1_SPECIES[] = {")
    n_species = 0
    for key in sorted(dex, key=lambda k: dex[k].get("num", 0)):
        e = dex[key]
        if not (1 <= e.get("num", 0) <= 151) or e.get("forme"):
            continue
        bs = e.get("baseStats") or e.get("base_stats")
        types = [t.upper() for t in e.get("types", [])]
        if not bs or not types or any(t not in ENUM for t in types):
            continue
        t1 = ENUM[types[0]]
        t2 = ENUM[types[1]] if len(types) > 1 else "Type::None"
        lines.append(f'    {{"{e.get("name", key)}", {t1}, {t2}, '
                     f'{bs["hp"]}, {bs["atk"]}, {bs["def"]}, {bs["spa"]}, {bs["spe"]}}},')
        n_species += 1
    lines += ["};", ""]

    lines.append("struct MoveEntry { const char* name; Type type; Category category; "
                 "int power, accuracy, fixed; Effect effect; int effect_chance; "
                 "int boost_stat, boost_stages; bool boost_target_foe; int boost_chance; "
                 "bool high_crit; int pp; bool recharge; int recoil_num, recoil_den; "
                 "bool ignore_immunity; int priority; bool skip_lastdamage; };")
    lines.append("inline const MoveEntry GEN1_MOVES[] = {")
    n_moves = 0
    for mid in sorted(moves, key=lambda k: moves[k].get("num", 0)):
        m = moves[mid]
        if type_to_enum(m.get("type", "")) == "Type::None":
            continue
        (name, typ, cat, power, accuracy, fixed, effect, ec, bstat, bstages, bfoe,
         bchance, high_crit, pp, recharge, rnum, rden, iimm, prio, skip) = translate_move(mid, m)
        lines.append(f'    {{"{name}", {typ}, {cat}, {power}, {accuracy}, {fixed}, '
                     f'{effect}, {ec}, {bstat}, {bstages}, {bfoe}, {bchance}, {high_crit}, {pp}, '
                     f'{recharge}, {rnum}, {rden}, {iimm}, {prio}, {skip}}},')
        n_moves += 1
    lines += ["};", "", "}  // namespace cinnabar"]
    return lines, n_species, n_moves


def main() -> None:
    gen1 = GenData.from_gen(1)
    ours, correct = ours_matrix(), correct_matrix(gen1)
    mismatches = [(a, d, ours[(a, d)], correct[(a, d)]) for a in ORDER for d in ORDER
                  if ours[(a, d)] != correct[(a, d)]]
    print("=== Type-chart audit ===")
    print("  no mismatches." if not mismatches else f"  {len(mismatches)} mismatch(es): {mismatches}")

    dex = gen1.pokedex or gen1.load_pokedex(1)
    moves = gen1.moves or gen1.load_moves(1)

    # Sanity peek at the move data shape (so key-name surprises surface immediately).
    sample = moves.get("bodyslam") or next(iter(moves.values()))
    print("\n=== Move data sample (bodyslam) ===")
    print(" ", {k: sample.get(k) for k in
                ("name", "type", "category", "basePower", "accuracy", "secondary", "boosts",
                 "status", "heal", "selfdestruct", "damage", "critRatio", "target", "pp")})
    print("\n=== Struggle raw ===")
    print(" ", moves.get("struggle"))
    print("=== base PP ===")
    print(" ", {mid: moves[mid].get("pp") for mid in
                ("recover", "softboiled", "amnesia", "bodyslam", "earthquake", "slash") if mid in moves})

    lines, n_species, n_moves = build_lines(correct, dex, moves)
    out = Path(__file__).resolve().parents[1] / "include" / "cinnabar" / "gen1_data.hpp"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}\n  {n_species} species, {n_moves} moves, corrected type chart")

    # Show how a few key moves translated, for eyeballing fidelity.
    print("\n=== Sample translations ===")
    for mid in ("bodyslam", "psychic", "amnesia", "swordsdance", "slash", "thunderwave",
                "recover", "seismictoss", "explosion", "growl", "blizzard", "icebeam", "hyperbeam",
                "substitute", "doubleedge", "takedown", "confuseray", "counter", "wrap"):
        if mid in moves:
            print(f"  {mid:12s} -> {translate_move(mid, moves[mid])}")


if __name__ == "__main__":
    main()
