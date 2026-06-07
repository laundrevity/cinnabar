# Cinnabar

Building and training an AI to play **Pokémon battles** — **Gen 1 OU**, chosen for its
mechanical simplicity (no abilities, no held items, small movepools) while still being a real
competitive format.

The project has two halves:

1. **A from-scratch Gen 1 battle engine in C++** (`engine/`), validated **bit-for-bit against
   [Pokémon Showdown](https://github.com/smogon/pokemon-showdown)** — same teams, same RNG
   seed, same choices produce identical state every turn, across tens of thousands of battles.
2. **A reinforcement-learning agent** (`agent/`) that trains on that engine **in-process** via
   PPO self-play at ~500 battles/sec — no Showdown server, no network in the training loop.

The engine-free decision core (state / policy / encoding / RL) sits behind adapters, so the
same agent code runs against either Showdown (via `poke-env`) or the custom C++ engine.

## Quickstart

**Train on the C++ engine** (the fast path):

```bash
# Build the engine + Python module (needs cmake + a C++20 compiler)
cd engine
cmake -S . -B build -DPython_EXECUTABLE=$(cd ../agent && uv run python -c 'import sys; print(sys.executable)')
cmake --build build
ctest --test-dir build --output-on-failure        # unit tests

cd ../agent
uv sync
uv run python smoke_engine.py                      # self-play on the engine (~1000 battles/s)
uv run python train_engine.py --opponent maxdamage --reward shaped --iters 150
#   -> agent goes from random to ~63% vs the max-damage baseline in a few minutes
```

**Verify engine fidelity vs Showdown** (needs the submodule built — `scripts/setup.sh`):

```bash
cd agent
uv run python ../engine/tools/trace_diff.py sweep 200   # diffs this engine vs Showdown, turn-for-turn
```

**Play / train via Showdown** (the original path):

```bash
scripts/setup.sh        # pull + build the Showdown submodule (needs network)
scripts/run-server.sh   # local server -> http://localhost:8000
cd agent && uv run python play.py    # accept human challenges in the browser
```

## Status

- **Engine:** full Gen 1 battles — 6v6 with switching, all statuses, stat stages, PP/Struggle,
  the 167-move table — matched bit-for-bit against Showdown. See [`engine/README.md`](engine/README.md).
- **Agent:** per-action PPO with self-play / league curriculum; beats the max-damage baseline
  and crushes random. Trains on the engine in-process (vectorized rollouts + batched updates).

## Layout

- `engine/` — the C++ Gen 1 engine, its pybind11 module, and the differential harness vs Showdown
- `agent/` — the RL agent (Python, `uv`): engine-free core + adapters for Showdown and the C++ engine
- `server/` — Pokémon Showdown (git submodule) + run scripts; used to play in-browser and as the fidelity oracle
- `teams/` — Gen 1 OU teams in Showdown export format
- `docs/` — roadmap and design notes
- `scripts/` — setup and server helpers

See [`CLAUDE.md`](CLAUDE.md) for the full project context and the architecture decisions
(including the deliberate reversal that led to building the custom engine).
