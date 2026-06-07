# engine/ — Cinnabar Gen 1 battle engine (C++)

A fast, from-scratch Gen 1 battle engine — a **deliberate engineering project** (see the
architecture note in `../CLAUDE.md`): build a faithful, fast engine in C++, with the RL agent
as a downstream consumer. Pure C++ core, no dependencies.

## Build & test (CMake)

```sh
cd engine
cmake -S . -B build                            # configure (needs cmake + a C++20 compiler)
cmake --build build                            # build the static lib + tests
ctest --test-dir build --output-on-failure     # run the unit tests
```

## Status — plays full battles (compiles + tested)

- 6-Pokémon teams + **switching**, with forced switch on faint and legal-action queries
  (`Battle::choices`), so it already matches the RL agent's move-or-switch action model
- **Status**: sleep (turn counter), freeze, paralysis (¼ Speed + 25% full-para), burn/poison
  (1/16 residual; burn halves Attack) — plus the moves that cause them (Thunder Wave, Sleep
  Powder, secondary effects on Body Slam/Blizzard/etc.)
- **Healing** (Recover/Soft-Boiled, Rest), **Reflect**, **Explosion/Self-Destruct**, fixed-damage
  moves (Seismic Toss)
- Gen 1 **damage formula**, **stat formula** (L100 maxed), **type chart** with the quirks
  (Ghost⇒Psychic = 0, Bug⇔Poison = 2×), type-based physical/special split, STAB, crit, immunities

Known simplifications to refine via differential testing: the 1/256 miss, exact crit rate,
freeze thaw, PP/Struggle, stat stages, Hyper Beam recharge, and a Showdown-bit-compatible RNG.

## Fidelity strategy (what keeps it honest)

1. **Differential testing against Showdown** is the source of truth: identical battle + RNG
   through this engine and Showdown's gen1 sim (our submodule), diffed turn-for-turn. (Next.)
2. **Generate static data from Showdown**, don't hand-type it. The species/move/type-chart data
   here is *provisional, hand-encoded* and must be replaced by data generated from Showdown's
   gen1 data files. Hand-typed game data is a fidelity-bug factory.

## Roadmap

1. **v0 foundation** — damage/stat formulas, type chart, 1v1 loop. *(done)*
2. **Full battles** — 6v6, switching, status + moves, healing, Reflect, Explosion. *(done)*
3. **Fidelity** — generate data from Showdown; build the differential-test harness; match
   Showdown's RNG; close the known simplifications above.
4. **Python bindings** — pybind11 → a new adapter yielding `agent/cinnabar`'s `BattleState`,
   so the existing engine-free RL core (state/policy/encoding/rl) runs on this engine unchanged.
5. **Scale** — vectorized, batched, in-process self-play + GPU training (the payoff).

## Layout

```
engine/
  CMakeLists.txt                # build (static lib + ctest)
  include/cinnabar/engine.hpp   # public API
  src/engine.cpp                # mechanics + (provisional) type chart
  tests/test_engine.cpp         # unit tests (hand-computed expectations)
```
