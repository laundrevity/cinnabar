# CLAUDE.md

Context for AI assistants working in this repo. Read this first.

## What this project is

Cinnabar is a project to **build and train an AI that plays Pokémon battles**, using
[Pokémon Showdown](https://github.com/smogon/pokemon-showdown) as the game engine and
interface. The progression is deliberately incremental:

- **v0** — a bot that picks a legal move at random, playable against a human in the browser.
- **later** — increasingly strong agents (heuristic baseline → reinforcement learning →
  self-play), each measured against the last.

## Scope (current)

- **Format: Gen 1 OU.** Chosen for mechanical simplicity — no abilities, no held items,
  small movepools, simpler type chart and status rules. This keeps the state/action space
  tractable while still being a real competitive format.
- Gen 1 OU *does* include team-building. We train over a **pool of fixed teams** (`teams/`,
  one chosen at random per battle via `agent/cinnabar/teams.py`) for generalization, and
  treat team-building (the agent constructing its own team) as a later, separable problem.

## Architecture decisions (and the reasoning — don't relitigate without cause)

1. **Custom C++ engine — now an explicit goal (`engine/`), a deliberate reversal of the
   original "don't".** We originally ruled out a custom simulator because the hard part of
   Pokémon is mechanics *fidelity*, not speed, and Showdown is the de facto reference. That
   reasoning still holds if the goal is "fastest path to a strong agent." But as a chosen
   **engineering project** (Conor, June 2026 — `@pkmn/engine` being pre-v0.1/unbuildable
   removed the off-the-shelf fast option), we are building a fast Gen 1 engine in C++. To
   avoid the fidelity trap it is (a) **scoped** to the moves/species our teams use, expanding
   outward, and (b) **validated by differential testing against Showdown** (identical battle +
   RNG, diff turn-for-turn). Showdown's role shifts from training backend to **fidelity oracle**.
   Static game data (type chart, base stats, movedex) should be **generated from Showdown data**,
   not hand-typed. The poke-env training path (`agent/`) still works and is unaffected.
2. **Throughput is not the early bottleneck.** The early blockers are state representation,
   reward shaping, and a training loop that converges. Run many headless Showdown instances
   in parallel for plenty of games/sec.
3. **If, and only if, profiling proves throughput is the wall, swap in
   [`@pkmn/engine`](https://github.com/pkmn/engine)** — a Zig engine purpose-built for ML
   self-play, ~1000× faster than Showdown's sim. It is pre-v0.1 and currently targets
   Gen 1/2, which aligns with our Gen 1 scope. This is a phase-3 optimization, not a
   foundation.
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
├── server/             # Pokémon Showdown lives here
│   └── pokemon-showdown/   # git submodule (added by scripts/setup.sh)
├── agent/              # Python agent (uv project)
│   ├── pyproject.toml      # deps (poke-env) + dev group (pytest, ruff)
│   ├── cinnabar/           # state.py + policy.py (engine-free) | showdown.py (poke-env adapter)
│   ├── play.py             # Phase 0: accept human challenges in Gen 1 OU
│   ├── smoke_test.py       # bot-vs-bot sanity check
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

## Conventions

- Showdown is a **submodule pinned to a commit** — bump it deliberately, not implicitly.
- `--no-security` is for **local use only**; never expose such a server publicly.
- Trained weights, replays, and logs are git-ignored; keep large artifacts out of the repo.
