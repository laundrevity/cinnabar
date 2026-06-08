"""Batch differential validator: run trace_diff sweeps for a list of moves vs Showdown.

For each move we set up a 1v1 where the attacker uses the move every turn and the target chips
back, then run an N-seed sweep through trace_diff.py. Reports which moves are already bit-for-bit
and which diverge (with the first divergence turn). Use it to audit move coverage before adding a
move to the training pool, and as a regression gate after engine changes.

Each entry is (move, attacker, target, target_move). Attacker/target default to a matchup where
the move is not type-immune and the battle runs long enough to exercise the mechanic many times.
Gen1 customgame lets any species use any move, so the species only matter for stats/typing.

Run from agent/:
    uv run python ../engine/tools/validate_moves.py [N]                 # audit the default list
    uv run python ../engine/tools/validate_moves.py [N] "Mega Drain" ...  # specific moves (defaults)
"""

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACE_DIFF = HERE / "trace_diff.py"

DEF_ATK, DEF_TGT, DEF_TGTM = "Snorlax", "Chansey", "Seismic Toss"

# Curated audit list for a realistic Gen 1 OU pool. Tuples may override (attacker, target,
# target_move); omitted fields fall back to the defaults above.
DEFAULT_MOVES: list[tuple] = [
    # --- expected already-correct (generic damage / status / boost / high-crit / fixed) ---
    ("Surf",),
    ("Fire Blast",),          # 30% burn secondary
    ("Bubble Beam",),         # 10% Spe drop secondary
    ("Aurora Beam",),         # 10% Atk drop secondary
    ("Razor Leaf",),          # high crit ratio
    ("Slash",),               # high crit ratio
    ("Sing", "Snorlax", "Chansey", "Soft-Boiled"),  # sleep (let target stall so sleep persists)
    ("Stun Spore",),          # paralysis
    ("Glare",),               # paralysis, ignores type immunity
    ("Amnesia",),             # +2 SpD/SpC boost
    ("Swords Dance",),        # +2 Atk boost
    ("Clamp", "Cloyster", "Chansey", "Soft-Boiled"),  # partial-trap (Trap effect)
    ("Fire Spin", "Moltres", "Chansey", "Soft-Boiled"),
    # --- suspected gaps (mechanic not modelled or mis-mapped) ---
    ("Mega Drain",),          # drain: heal half damage dealt
    ("Absorb",),
    ("Leech Life",),
    ("Super Fang",),          # special: damage = half target current HP
    ("Pin Missile",),         # multi-hit 2-5
    ("Double Kick",),         # multi-hit fixed 2
    ("Light Screen", "Snorlax", "Chansey", "Psychic"),  # opponent must hit specially to see halving
    ("Leech Seed", "Snorlax", "Chansey", "Soft-Boiled"),  # residual drain; target stalls so it persists
    ("Toxic", "Snorlax", "Chansey", "Soft-Boiled"),      # gen1 escalating poison (residualdmg counter)
    ("Psywave",),             # random damage
    ("Disable",),             # disable a random foe move for N turns (consumes RNG -> desyncs if unmodelled)
    # NOTE: Haze is intentionally omitted — it's a no-op (and thus a false PASS) unless stat boosts or
    # status are present. Validate it manually with a boost set up first (e.g. Amnesia, then Haze).
]


def parse_spec(spec: tuple) -> tuple[str, str, str, str]:
    move = spec[0]
    atk = spec[1] if len(spec) > 1 else DEF_ATK
    tgt = spec[2] if len(spec) > 2 else DEF_TGT
    tgtm = spec[3] if len(spec) > 3 else DEF_TGTM
    return move, atk, tgt, tgtm


def run_one(move: str, atk: str, tgt: str, tgtm: str, n: int) -> tuple[bool, str]:
    env = dict(os.environ)
    env["CINNABAR_P1_SPECIES"] = atk
    env["CINNABAR_P1_MOVE"] = move
    env["CINNABAR_P2_SPECIES"] = tgt
    env["CINNABAR_P2_MOVE"] = tgtm
    # Drop any team-level overrides so only the single-species matchup is used.
    for k in ("CINNABAR_P1_TEAM", "CINNABAR_P2_TEAM"):
        env.pop(k, None)
    out = subprocess.run(
        [sys.executable, str(TRACE_DIFF), "sweep", str(n)],
        capture_output=True, text=True, env=env,
    )
    if out.returncode != 0:
        return False, f"ERROR: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else 'sweep failed'}"
    text = out.stdout
    m = re.search(r"sweep:\s+(\d+)/(\d+)", text)
    if not m:
        return False, "no sweep result parsed"
    passed, total = int(m.group(1)), int(m.group(2))
    ok = passed == total
    detail = f"{passed}/{total}"
    if not ok:
        tm = re.search(r"turn (\d+):", text)
        if tm:
            detail += f"  first divergence ~turn {tm.group(1)}"
    return ok, detail


def main() -> int:
    args = sys.argv[1:]
    n = 100
    if args and args[0].isdigit():
        n = int(args[0])
        args = args[1:]
    specs = [(m,) for m in args] if args else DEFAULT_MOVES

    print(f"validating {len(specs)} move(s), {n} seeds each\n")
    print(f"{'move':<16} {'matchup':<26} {'result':<10} detail")
    print("-" * 78)
    fails = []
    for spec in specs:
        move, atk, tgt, tgtm = parse_spec(spec)
        ok, detail = run_one(move, atk, tgt, tgtm, n)
        mark = "PASS" if ok else "FAIL"
        print(f"{move:<16} {f'{atk} vs {tgt}':<26} {mark:<10} {detail}")
        if not ok:
            fails.append(move)
    print("-" * 78)
    if fails:
        print(f"\n{len(fails)} need work: {', '.join(fails)}")
    else:
        print("\nall validated bit-for-bit ✓")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
