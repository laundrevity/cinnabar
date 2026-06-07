"""Exact trace diff: our C++ engine vs Showdown's gen1 sim, same seed + choices.

1v1 Tauros vs Snorlax, both "move 1" (Earthquake) every turn. Identical Gen-5 LCG
seed in both. Prints the first turn where the two engines disagree on HP/status —
that's the first fidelity bug (almost certainly an RNG-call-order mismatch to start).

Needs: built module (engine/build), Node, and the built Showdown submodule.
Run from agent/:  uv run python ../engine/tools/trace_diff.py [s0 s1 s2 s3]
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "build"))
import cinnabar_engine as ce  # noqa: E402

# Matchup configurable via env (mirrors ref_trace.js, which inherits this environment).
# CINNABAR_P{1,2}_TEAM = comma-separated species (each gets CINNABAR_P{1,2}_MOVE).
def _team(team_env, sp_env, move_env):
    species = (os.environ.get(team_env) or os.environ.get(sp_env) or "Tauros").split(",")
    move = os.environ.get(move_env, "Earthquake")
    return [(s.strip(), [move]) for s in species]


P1 = _team("CINNABAR_P1_TEAM", "CINNABAR_P1_SPECIES", "CINNABAR_P1_MOVE")
P2 = _team("CINNABAR_P2_TEAM", "CINNABAR_P2_SPECIES", "CINNABAR_P2_MOVE")


VOL = bool(os.environ.get("CINNABAR_VOL"))  # enable voluntary switches (mirrors ref_trace.js)


def choose(b, player, counter):
    """Forced switch (only switches offered) -> lowest-index alive teammate. Else, with
    voluntary switching on, switch on the scheduled counter; otherwise attack with move 0."""
    cs = b.choices(player)
    if cs and all(c.kind == ce.ChoiceKind.Switch for c in cs):
        return min(cs, key=lambda c: c.index)
    if VOL and ((player == 0 and counter % 5 == 1) or (player == 1 and counter % 5 == 3)):
        switches = [c for c in cs if c.kind == ce.ChoiceKind.Switch]
        if switches:
            return min(switches, key=lambda c: c.index)
    for c in cs:
        if c.kind == ce.ChoiceKind.Move and c.index == 0:
            return c
    for c in cs:
        if c.kind == ce.ChoiceKind.Move:  # Struggle (-1) or any available move
            return c
    return cs[0]


def our_trace(seed_u64):
    b = ce.make_battle(P1, P2, seed_u64)
    # Showdown marks a fainted Pokémon's status as 'fnt'; mirror that in the snapshot.
    def st(pl):
        return "fnt" if b.active_hp(pl) <= 0 else b.active_status(pl)
    tr, guard, counter = [], 0, 0
    while b.result() == ce.Result.Ongoing and guard < 2000:
        guard += 1
        b.step(choose(b, 0, counter), choose(b, 1, counter))
        counter += 1
        tr.append({
            "p1_sp": b.active_species(0), "p1_hp": b.active_hp(0), "p1_maxhp": b.active_max_hp(0), "p1_status": st(0),
            "p2_sp": b.active_species(1), "p2_hp": b.active_hp(1), "p2_maxhp": b.active_max_hp(1), "p2_status": st(1),
        })
    winner = {ce.Result.P1Win: "p1", ce.Result.P2Win: "p2"}.get(b.result())
    return {"winner": winner, "trace": tr}


def showdown_trace(words):
    out = subprocess.run(
        ["node", str(HERE / "ref_trace.js"), *map(str, words)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print("Showdown sim failed:\n" + out.stderr, file=sys.stderr)
        sys.exit(1)
    return json.loads(out.stdout)


def words_to_u64(w):
    return (w[0] << 48) | (w[1] << 32) | (w[2] << 16) | w[3]


def run_sweep(n):
    out = subprocess.run(
        ["node", str(HERE / "ref_trace.js"), "--sweep", str(n)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print("Showdown sim failed:\n" + out.stderr, file=sys.stderr)
        sys.exit(1)
    sweep = json.loads(out.stdout)["sweep"]

    passed, first_fail = 0, None
    for entry in sweep:
        ot = our_trace(words_to_u64(entry["words"]))["trace"]
        tt = entry["trace"]
        if len(ot) == len(tt) and all(a == b for a, b in zip(ot, tt)):
            passed += 1
        elif first_fail is None:
            first_fail = (entry["words"], ot, tt)

    print(f"sweep: {passed}/{len(sweep)} battles identical")
    if first_fail:
        words, ot, tt = first_fail
        m = min(len(ot), len(tt))
        j = next((i for i in range(m) if ot[i] != tt[i]), m)
        print(f"\nfirst failing seed words {words}:")
        for k in range(max(0, j - 1), min(m, j + 2)):
            mark = "  <-- differs" if k < m and ot[k] != tt[k] else ""
            print(f"  turn {k + 1}:")
            print(f"    ours     {ot[k]}{mark}")
            print(f"    showdown {tt[k]}")
        if len(ot) != len(tt):
            print(f"  (trace lengths differ: ours {len(ot)} vs showdown {len(tt)})")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "sweep":
        run_sweep(int(sys.argv[2]) if len(sys.argv) >= 3 else 100)
        return

    words = [int(x) for x in sys.argv[1:5]] if len(sys.argv) >= 5 else [1, 2, 3, 4]
    seed_u64 = (words[0] << 48) | (words[1] << 32) | (words[2] << 16) | words[3]

    ours = our_trace(seed_u64)
    theirs = showdown_trace(words)
    ot, tt = ours["trace"], theirs["trace"]

    print(f"seed words {words} (u64={seed_u64})")
    print(f"ours:     {len(ot):3d} turns, winner {ours['winner']}")
    print(f"showdown: {len(tt):3d} turns, winner {theirs['winner']}")

    n = min(len(ot), len(tt))
    first = next((i for i in range(n) if ot[i] != tt[i]), None)
    if first is None and len(ot) == len(tt):
        print("\nIDENTICAL — engine matches Showdown bit-for-bit on this battle. ✓")
        return
    j = first if first is not None else n
    print(f"\nfirst divergence around turn {j + 1 if first is not None else n}:")
    for k in range(max(0, j - 1), min(n, j + 2)):
        mark = "  <-- differs" if ot[k] != tt[k] else ""
        print(f"  turn {k + 1}:")
        print(f"    ours     {ot[k]}{mark}")
        print(f"    showdown {tt[k]}")
    if first is None:
        print(f"  (then trace lengths differ: ours {len(ot)} vs showdown {len(tt)})")


if __name__ == "__main__":
    main()
