"""Exact trace diff: our C++ engine vs Showdown's gen1 sim, same seed + choices.

1v1 Tauros vs Snorlax, both "move 1" (Earthquake) every turn. Identical Gen-5 LCG
seed in both. Prints the first turn where the two engines disagree on HP/status —
that's the first fidelity bug (almost certainly an RNG-call-order mismatch to start).

Needs: built module (engine/build), Node, and the built Showdown submodule.
Run from agent/:  uv run python ../engine/tools/trace_diff.py [s0 s1 s2 s3]
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "build"))
import cinnabar_engine as ce  # noqa: E402

P1 = [("Tauros", ["Earthquake"])]
P2 = [("Snorlax", ["Earthquake"])]


def move0(b, player):
    for c in b.choices(player):
        if c.kind == ce.ChoiceKind.Move and c.index == 0:
            return c
    return b.choices(player)[0]


def our_trace(seed_u64):
    b = ce.make_battle(P1, P2, seed_u64)
    tr, guard = [], 0
    while b.result() == ce.Result.Ongoing and guard < 2000:
        guard += 1
        b.step(move0(b, 0), move0(b, 1))
        tr.append({
            "turn": b.turn,
            "p1_hp": b.active_hp(0), "p1_maxhp": b.active_max_hp(0), "p1_status": b.active_status(0),
            "p2_hp": b.active_hp(1), "p2_maxhp": b.active_max_hp(1), "p2_status": b.active_status(1),
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


def main():
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
