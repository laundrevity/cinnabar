"""Smoke test for the C++ engine RL adapter.

Builds a BattleState from the engine, runs RandomPolicy self-play battles entirely
in-process on the C++ engine (no Showdown server), and reports throughput.

    cd agent && uv run python smoke_engine.py
"""

from __future__ import annotations

import time

from cinnabar.engine_cpp import StaticData, build_state, play_battle  # inserts engine/build on sys.path
import cinnabar_engine as ce  # noqa: E402
from cinnabar.policy import RandomPolicy  # noqa: E402

TEAM_A = [
    ("Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]),
    ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
    ("Alakazam", ["Psychic", "Thunder Wave", "Recover", "Seismic Toss"]),
]
TEAM_B = [
    ("Starmie", ["Thunderbolt", "Ice Beam", "Recover", "Thunder Wave"]),
    ("Chansey", ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
    ("Exeggutor", ["Psychic", "Sleep Powder", "Explosion", "Body Slam"]),
]


def main() -> None:
    static = StaticData(1)

    # 1) Show a BattleState built from the engine.
    spec1 = [(s, list(m)) for s, m in TEAM_A]
    spec2 = [(s, list(m)) for s, m in TEAM_B]
    b = ce.make_battle(spec1, spec2, 1)
    st = build_state(b, 0, spec1, static, "demo")
    print("Sample BattleState (p1):")
    print(f"  active     : {st.active.species} hp={st.active.hp_fraction:.2f} "
          f"types={st.active.types} spe={st.active.speed}")
    print(f"  opp active : {st.opponent_active.species}")
    print(f"  team       : {[m.species for m in st.team]}")
    print(f"  actions ({len(st.available_actions)}):")
    for a in st.available_actions:
        if a.type.value == "move":
            print(f"    move   {a.label:14s} bp={a.base_power:5.0f} type={a.move_type:8s} x{a.type_multiplier}")
        else:
            print(f"    switch {a.species:14s} hp={a.target_hp_fraction:.2f} incoming x{a.incoming_multiplier}")

    # 2) RandomPolicy self-play; count outcomes + throughput.
    p1, p2 = RandomPolicy(1), RandomPolicy(2)
    tally = {ce.Result.P1Win: 0, ce.Result.P2Win: 0, ce.Result.Tie: 0, ce.Result.Ongoing: 0}
    n = 2000
    t0 = time.time()
    for i in range(n):
        tally[play_battle(p1, p2, TEAM_A, TEAM_B, static, seed=i + 1).result()] += 1
    dt = time.time() - t0
    print(f"\n{n} RandomPolicy self-play battles in {dt:.2f}s = {n / dt:,.0f} battles/sec")
    print(f"  p1 {tally[ce.Result.P1Win]} | p2 {tally[ce.Result.P2Win]} | "
          f"ties {tally[ce.Result.Tie]} | unfinished {tally[ce.Result.Ongoing]}")


if __name__ == "__main__":
    main()
