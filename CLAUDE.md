# CLAUDE.md

Context for AI assistants working in this repo. Read this first.

## What this project is

Cinnabar is a project to **build and train an AI that plays Pokémon battles**, using
[Pokémon Showdown](https://github.com/smogon/pokemon-showdown) as the game engine and
interface. The progression is deliberately incremental:

- **v0** — a bot that picks a legal move at random, playable against a human in the browser.
- **later** — increasingly strong agents (heuristic baseline → reinforcement learning →
  self-play), each measured against the last.

**Status (current):** all of the above is built. The agent (Phases 0–3) trains via per-action
PPO with self-play / league curriculum. The custom C++ engine (`engine/`) is **validated
bit-for-bit against Showdown** for full 6v6 Gen 1 battles, and the RL agent trains on it
**in-process** at ~500 battles/sec (`agent/train_engine.py`) — random → beats max-damage in
minutes. Remaining work is open-ended ML iteration (curriculum, breadth, tuning), not building.

## Scope (current)

- **Format: Gen 1 OU.** Chosen for mechanical simplicity — no abilities, no held items,
  small movepools, simpler type chart and status rules. This keeps the state/action space
  tractable while still being a real competitive format.
- Gen 1 OU *does* include team-building. We train over a **pool of fixed teams** (`teams/`,
  one chosen at random per battle via `agent/cinnabar/teams.py`) for generalization. Team-building
  (the agent constructing its own team) was the deferred "later, separable problem"; the first cut
  is now `agent/evolve_teams.py` — coevolutionary search that evolves winning teams with the trained
  agent as pilot (a learned drafter / PSRO are the later, more ambitious steps).

## Architecture decisions (and the reasoning — don't relitigate without cause)

1. **Custom C++ engine — now an explicit goal (`engine/`), a deliberate reversal of the
   original "don't".** We originally ruled out a custom simulator because the hard part of
   Pokémon is mechanics *fidelity*, not speed, and Showdown is the de facto reference. That
   reasoning still holds if the goal is "fastest path to a strong agent." But as a chosen
   **engineering project** (Conor, June 2026 — `@pkmn/engine` being pre-v0.1/unbuildable
   removed the off-the-shelf fast option), we **built** a fast Gen 1 engine in C++ (`engine/`).
   To avoid the fidelity trap it is (a) **scoped** to the moves/species our teams use, expanding
   outward, and (b) **validated by differential testing against Showdown** (identical battle +
   RNG, diff turn-for-turn) — now passing **bit-for-bit across full 6v6 battles**, tens of
   thousands of seeds. Showdown's role shifted from training backend to **fidelity oracle**.
   Static game data (type chart, base stats, movedex) is **generated from Showdown** (`tools/
   gen_data.py`), not hand-typed. The poke-env training path (`agent/`) still works.
2. **Throughput was not the early bottleneck** — state representation, reward shaping, and a
   converging loop were. Those solved (Phases 0–3 on Showdown), the custom engine then became
   the throughput path: the RL loop now runs **in-process on the C++ engine** (vectorized
   rollouts + batched PPO), ~500 battles/sec vs the Showdown-websocket path's ~1–2/sec.
3. **`@pkmn/engine` was evaluated and abandoned** (pre-v0.1, unbuildable at the time), which is
   what motivated decision #1. The custom engine is the throughput foundation instead.
4. **Agent language: Python, managed with [`uv`](https://docs.astral.sh/uv/)** (not pip/venv).
   The ML ecosystem (Gymnasium, PyTorch, stable-baselines3) and
   [`poke-env`](https://poke-env.readthedocs.io/) — the Showdown client + battle-state
   tracker + agent API, which ships a `RandomPlayer` and a Gym wrapper — live here.
5. **poke-env is kept behind an adapter** (`agent/cinnabar/showdown.py`). The decision core
   (`agent/cinnabar/state.py`, `policy.py`) reasons over our own `BattleState`/`Action` types
   and has no engine dependency, so agents can be retargeted (e.g. to `@pkmn/engine`) by
   rewriting one file. Every agent — random, heuristic, RL — implements `Policy.select_action`.

## Repo layout

```
cinnabar/
├── CLAUDE.md            # this file
├── README.md           # human-facing intro + quickstart
├── docs/roadmap.md     # the phased plan in detail
├── engine/             # custom C++ Gen 1 engine (validated bit-for-bit vs Showdown)
│   ├── include/cinnabar/engine.hpp + gen1_data.hpp (GENERATED) | src/engine.cpp | tests/
│   ├── bindings/bind.cpp   # pybind11 module: import cinnabar_engine
│   └── tools/              # gen_data.py (codegen) | ref_trace.js + trace_diff.py (fidelity harness)
├── server/             # Pokémon Showdown lives here
│   └── pokemon-showdown/   # git submodule (added by scripts/setup.sh); also the fidelity oracle
├── agent/              # Python agent (uv project)
│   ├── pyproject.toml      # deps (poke-env) + dev group (pytest, ruff)
│   ├── cinnabar/           # state.py + policy.py + encoding.py + rl/ (engine-free core)
│   │                       #   showdown.py (poke-env adapter) | engine_cpp.py (C++ engine adapter)
│   ├── play.py             # Phase 0: accept human challenges in Gen 1 OU
│   ├── train.py            # PPO training via Showdown (poke-env)
│   ├── train_engine.py     # PPO training in-process on the C++ engine (vectorized); --clauses for OU Sleep/Freeze Clause
│   ├── evolve_teams.py     # coevolutionary team optimizer (team construction): evolve winning teams, agent pilots both sides
│   ├── ladder.py           # Elo ladder over baselines + checkpoints (--clauses to rate under OU clauses)
│   ├── pad_checkpoint.py   # zero-pad/frame-replicate old checkpoints to the current GLOBAL_DIM (fair warm-start)
│   ├── smoke_test.py / smoke_engine.py  # bot-vs-bot sanity checks (Showdown / C++ engine)
│   └── tests/              # pytest for the engine-free core
├── teams/              # Gen 1 OU teams in Showdown export format
└── scripts/
    ├── setup.sh        # add submodule + npm install + build
    └── run-server.sh   # start a local server (--no-security) on :8000
```

## Common commands

- **Set up Showdown:** `scripts/setup.sh` (needs network; run locally, not in a sandbox)
- **Run local server:** `scripts/run-server.sh` → http://localhost:8000
  - WebSocket endpoint: `ws://localhost:8000/showdown/websocket`
- **Play in browser:** start the server, open the printed URL, pick Gen 1 OU.
- **Set up the agent:** `cd agent && uv sync`
- **Smoke test (bot vs bot):** `cd agent && uv run python smoke_test.py` (server running)
- **Play the bot (random vs human):** `cd agent && uv run python play.py`
- **Tests / lint:** `cd agent && uv run pytest` · `uv run ruff check`

### Engine (C++) + engine-backed training

- **Build engine + module:** `cd engine && cmake -S . -B build -DPython_EXECUTABLE=$(cd ../agent && uv run python -c 'import sys; print(sys.executable)') && cmake --build build`
- **Engine unit tests:** `ctest --test-dir build --output-on-failure`
- **Regenerate static data from Showdown:** `cd agent && uv run python ../engine/tools/gen_data.py`
- **Fidelity harness (engine vs Showdown):** `cd agent && uv run python ../engine/tools/trace_diff.py sweep 200` (needs the submodule built; env vars `CINNABAR_P{1,2}_{SPECIES,TEAM,MOVE}`, `CINNABAR_VOL=1`)
- **Self-play smoke on the engine:** `cd agent && uv run python smoke_engine.py`
- **Train on the engine:** `cd agent && uv run python train_engine.py --opponent maxdamage --reward shaped` (`--smoke` for a tiny run; `--opponent self` for self-play)
- **Train with OU clauses:** add `--clauses` (Sleep + Freeze Clause — a 2nd foe-inflicted sleep/freeze fails). Modeled as a per-`Side` flag, **default off** so the bit-for-bit harness (clause-free `gen1customgame`) is untouched; `ladder.py --clauses` rates under the same rule. Without it the agent over-values sleep (it can sleep your whole team in training). GLOBAL_DIM gained 2 clause-perception features — warm-start old nets with `pad_checkpoint.py`.
- **Evolve teams (team construction):** `cd agent && uv run python evolve_teams.py --ckpt models_clauses/pg_best.pt --pop 24 --gens 30 --clauses --out evolved` — coevolution: the agent pilots both sides, teams compete vs the population, top teams written as Showdown `.txt`. The pilot ckpt must match the current GLOBAL_DIM (use `pad_checkpoint.py` on older ones).

## Conventions

- Showdown is a **submodule pinned to a commit** — bump it deliberately, not implicitly.
- `--no-security` is for **local use only**; never expose such a server publicly.
- Trained weights, replays, and logs are git-ignored; keep large artifacts out of the repo.
