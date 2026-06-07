# engine/ — Cinnabar Gen 1 battle engine (C++)

A fast, from-scratch Gen 1 battle engine — a **deliberate engineering project** (see the
architecture note in `../CLAUDE.md`): build a faithful, fast engine in C++, with the RL agent
as a downstream consumer. Pure C++ core, no dependencies.

It is **validated bit-for-bit against Pokémon Showdown** (same teams, same RNG seed, same
choices → identical state every turn) across tens of thousands of battles, and the RL agent
trains on it in-process at ~500 battles/sec (`../agent/train_engine.py`).

## Build & test (CMake)

```sh
cd engine
cmake -S . -B build                            # configure (needs cmake + a C++20 compiler)
cmake --build build                            # static lib + ctest + the pybind module
ctest --test-dir build --output-on-failure     # run the unit tests
```

To build the Python module against the agent's interpreter:

```sh
cmake -S . -B build -DPython_EXECUTABLE=$(cd ../agent && uv run python -c 'import sys; print(sys.executable)')
cmake --build build      # -> build/cinnabar_engine.<...>.so
```

## Status — full Gen 1 battles, validated vs Showdown

Every mechanic below is matched **bit-for-bit** against Showdown's own gen1 sim via the
differential harness (`tools/ref_trace.js` + `tools/trace_diff.py`):

- **RNG**: Showdown's exact Gen 5 LCG, with the call sequence (accuracy → crit → damage roll →
  secondary) aligned to Showdown's.
- **Damage**: the full Gen 1 formula incl. the ≥256 **stat rollover**, crit (level-doubling,
  ignores boosts/screens/burn), STAB, and per-type effectiveness applied **in order with
  integer flooring** (the dual-type rounding quirk).
- **Type chart** with the Gen 1 quirks (Ghost⇒Psychic = 0, Bug⇔Poison = 2×), type-based
  physical/special split, immunities. **Stat formula** (L100, max StatExp).
- **Accuracy** incl. the 1/256 miss; **crit** from base Speed (high crit-ratio moves too).
- **Status**: sleep, freeze, paralysis (63/256 full-para + Speed drop), burn/poison
  (1/16 after the mon's own move, skipped on a KO turn), and the moves that cause them — incl.
  the gen1 "a Normal move can't 2ndary-status a Normal-type" rule.
- **Stat stages**: the boost table, `modifiedStats`, burn/paralysis drops re-applied on
  switch-in, crit ignoring boosts (Amnesia / Swords Dance / Agility / Psychic's −Special).
- **PP + Struggle** (max PP = base×8/5; Struggle with accuracy roll + floored/capped recoil).
- **Switching**: 6v6, forced switch on faint, voluntary switch (resolves before moves),
  stat stages reset on switch, mid-turn battle end. **Speed ties** (the steady-state shuffle
  frame), **Self-Destruct/Explosion** (faints the user even on a miss).
- The full **167-move table** is generated from Showdown (`tools/gen_data.py`).

**Documented simplifications** (deliberate, narrow): Toxic's escalating counter is treated as
flat poison; bit-exact RNG on *speed-tie-and-switch* turns isn't reproduced (a fair coin
either way — no gameplay effect); and moves whose *special* mechanic isn't modeled yet
(Counter, Substitute, multi-hit, two-turn, partial-trap, confusion, Light Screen, …) carry
correct power/type/accuracy but their special behavior is a no-op.

## Differential harness (what keeps it honest)

`tools/trace_diff.py` runs the same battle through this engine and Showdown's gen1 sim with an
identical Gen-5 seed and choice script, and diffs the per-turn state. `sweep N` runs N seeds.

```sh
cd ../agent
uv run python ../engine/tools/trace_diff.py sweep 200            # 1v1, Earthquake mirror
CINNABAR_P1_MOVE="Body Slam" CINNABAR_P2_MOVE=Earthquake \
  uv run python ../engine/tools/trace_diff.py sweep 200          # any move matchup
CINNABAR_VOL=1 CINNABAR_P1_TEAM="Tauros,Alakazam,Chansey" \
  CINNABAR_P2_TEAM="Snorlax,Starmie,Exeggutor" \
  uv run python ../engine/tools/trace_diff.py sweep 100          # 6v6 + switching
```

Static data is generated from Showdown, never hand-typed: `tools/gen_data.py` pulls poke-env's
Gen 1 tables and emits `include/cinnabar/gen1_data.hpp` (type chart + 151 species + 167 moves).

## Python bindings + RL adapter

`bindings/bind.cpp` exposes `make_battle`, `Battle.choices/step/result`, `team_state`,
`must_switch`, and per-active accessors. The agent's `cinnabar/engine_cpp.py` adapter turns
that into the engine-free RL core's `BattleState`/`Action`, so the existing
state/policy/encoding/PPO code trains on this engine unchanged:

```sh
cd ../agent
uv run python smoke_engine.py                  # RandomPolicy self-play on the engine (~1000/s)
uv run python train_engine.py --smoke          # vectorized PPO on the engine
```

## Layout

```
engine/
  CMakeLists.txt                    # build (static lib + ctest + pybind module)
  include/cinnabar/engine.hpp       # public API
  include/cinnabar/gen1_data.hpp    # GENERATED: type chart + 151 species + 167 moves (do not edit)
  src/engine.cpp                    # mechanics; data sourced from gen1_data.hpp
  tests/test_engine.cpp             # unit tests (hand-computed expectations)
  tools/gen_data.py                 # regenerate gen1_data.hpp from Showdown (poke-env)
  tools/ref_trace.js                # Showdown reference trace (drives the submodule sim)
  tools/trace_diff.py               # differential harness: this engine vs Showdown
  bindings/bind.cpp                 # pybind11 module (import cinnabar_engine)
  bindings/smoke.py                 # Python-drives-C++ smoke + throughput
```
