"""Differential fidelity harness (v1, distributional).

Plays many random-vs-random Gen 1 OU mirror battles in BOTH our C++ engine and
Showdown (same team), and compares battle-length distributions + win rates. Big
divergences flag gross mechanics differences.

This is coarse by design: it cannot catch subtle per-turn bugs (that needs an
RNG-synced turn-for-turn diff), but it surfaces gross issues and the impact of our
known simplifications (no 1/256 miss, approximate crit rate, sleep distribution,
no freeze thaw / Hyper Beam recharge). The stronger v2 is a deterministic damage
cross-check against poke_env.calc.damage_calc_gen1_2.calculate_damage_gen12.

Needs: the built module (engine/build), poke-env, and a running Showdown server.
Run from agent/:  uv run python ../engine/tools/diff_harness.py [N]
"""

import asyncio
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import cinnabar_engine as ce  # noqa: E402

from poke_env import RandomPlayer  # noqa: E402

# One team, in both forms. Uses only moves the engine's move() table currently has.
TEAM = [
    ("Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]),
    ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
    ("Chansey", ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
    ("Exeggutor", ["Psychic", "Sleep Powder", "Explosion", "Body Slam"]),
    ("Starmie", ["Thunderbolt", "Ice Beam", "Recover", "Thunder Wave"]),
    ("Alakazam", ["Psychic", "Seismic Toss", "Thunder Wave", "Recover"]),
]
TEAM_STR = """
Tauros
- Body Slam
- Earthquake
- Blizzard
- Hyper Beam

Snorlax
- Body Slam
- Earthquake
- Hyper Beam
- Rest

Chansey
- Ice Beam
- Thunderbolt
- Thunder Wave
- Soft-Boiled

Exeggutor
- Psychic
- Sleep Powder
- Explosion
- Body Slam

Starmie
- Thunderbolt
- Ice Beam
- Recover
- Thunder Wave

Alakazam
- Psychic
- Seismic Toss
- Thunder Wave
- Recover
"""


def pct(xs, p):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_engine(n):
    turns, p1w, ties, fin = [], 0, 0, 0
    for seed in range(n):
        b = ce.make_battle(TEAM, TEAM, seed)
        steps = 0
        while b.result() == ce.Result.Ongoing and steps < 2000:
            b.step(random.choice(b.choices(0)), random.choice(b.choices(1)))
            steps += 1
        r = b.result()
        if r == ce.Result.Ongoing:
            continue  # hit the safety cap; skip
        fin += 1
        turns.append(b.turn)
        if r == ce.Result.P1Win:
            p1w += 1
        elif r == ce.Result.Tie:
            ties += 1
    return turns, p1w, ties, fin


async def run_showdown(n):
    common = dict(battle_format="gen1ou", team=TEAM_STR, max_concurrent_battles=10, log_level=40)
    p1, p2 = RandomPlayer(**common), RandomPlayer(**common)
    await p1.battle_against(p2, n_battles=n)
    turns, p1w, ties, fin = [], 0, 0, 0
    for b in p1.battles.values():
        if not b.finished:
            continue
        fin += 1
        turns.append(b.turn)
        if b.won is True:
            p1w += 1
        elif b.won is None:
            ties += 1
    return turns, p1w, ties, fin


def summary(name, turns, p1w, ties, fin):
    print(f"{name:11s} | battles {fin:4d} | p1 win {100 * p1w / max(fin, 1):5.1f}% | "
          f"ties {ties:3d} | turns mean {statistics.mean(turns):5.1f} "
          f"median {statistics.median(turns):5.1f} p10 {pct(turns, 10):4.0f} p90 {pct(turns, 90):4.0f}")


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"Running {n} random mirror battles in each engine (Showdown needs the local server up)...\n")
    e = run_engine(n)
    s = await run_showdown(n)
    summary("C++ engine", *e)
    summary("Showdown", *s)
    em, sm = statistics.mean(e[0]), statistics.mean(s[0])
    print(f"\nmean battle length: engine {em:.1f} vs Showdown {sm:.1f} ({100 * (em - sm) / sm:+.0f}% vs oracle)")
    print("Sanity: both p1 win rates ~50% (mirror). A large turn-length gap flags mechanics\n"
          "differences — some expected from our known simplifications; investigate big ones.")


if __name__ == "__main__":
    asyncio.run(main())
