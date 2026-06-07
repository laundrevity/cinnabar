"""Smoke test: drive the C++ engine from Python via the pybind11 module.

Build the module first (from engine/):
    cmake -S . -B build -DPython3_EXECUTABLE=$(cd ../agent && uv run python -c 'import sys;print(sys.executable)')
    cmake --build build
Then run with that same interpreter:
    cd ../agent && uv run python ../engine/bindings/smoke.py

It plays random battles entirely in C++ and reports battles/sec — the throughput that
justified building the engine in the first place.
"""

import random
import sys
import time
from pathlib import Path

# The built module (cinnabar_engine.*.so) lands in engine/build/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import cinnabar_engine as ce  # noqa: E402

TEAM = [
    ("Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]),
    ("Chansey", ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
    ("Snorlax", ["Body Slam", "Earthquake", "Explosion", "Rest"]),
    ("Exeggutor", ["Psychic", "Sleep Powder", "Explosion", "Body Slam"]),
    ("Starmie", ["Thunderbolt", "Ice Beam", "Recover", "Thunder Wave"]),
    ("Alakazam", ["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
]


def play_one(seed: int) -> tuple:
    b = ce.make_battle(TEAM, TEAM, seed)
    turns = 0
    while b.result() == ce.Result.Ongoing and turns < 2000:
        c1 = random.choice(b.choices(0))
        c2 = random.choice(b.choices(1))
        b.step(c1, c2)
        turns += 1
    return b.result(), b.turn


def main() -> None:
    print("one battle, narrated start:", play_one(1))

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    for i in range(200):  # warmup
        play_one(i)
    t0 = time.perf_counter()
    for i in range(n):
        play_one(i)
    secs = time.perf_counter() - t0
    print(f"{n} battles in {secs:.2f}s = {round(n / secs)} battles/sec "
          f"(Python driving the C++ engine)")


if __name__ == "__main__":
    main()
