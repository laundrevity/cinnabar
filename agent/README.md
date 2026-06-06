# agent/

The AI agent that plays battles. **Language: Python** (via
[`poke-env`](https://poke-env.readthedocs.io/), which handles the Showdown
WebSocket protocol and battle-state tracking).

## Design: poke-env behind an adapter

The package is split so the decision-making core never touches poke-env:

```
pyproject.toml # uv project: deps (poke-env, torch) + dev group (pytest, ruff)
cinnabar/
  state.py     # our BattleState + Action types        (no poke-env)
  policy.py    # Policy ABC + RandomPolicy + MaxDamagePolicy  (no poke-env)
  encoding.py  # BattleState -> feature vectors         (dependency-free)
  showdown.py  # PolicyPlayer: the poke-env adapter     (the only poke-env code)
  rl/          # PyTorch agent: net.py, agent.py (PGPolicy), returns.py (torch-free)
play.py        # Phase 0: accept human challenges in Gen 1 OU
smoke_test.py  # bot-vs-bot sanity check, watchable in the browser
evaluate.py    # win-rate harness: policy A vs policy B over N games
train.py       # Phase 2: REINFORCE-with-baseline training loop
tests/         # pytest for the engine-free core
```

Every future agent (heuristic, RL, self-play) implements `Policy.select_action`.
If we ever swap poke-env for a faster engine, only `showdown.py` changes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a local Showdown server
(`../scripts/run-server.sh`).

```bash
cd agent
uv sync            # creates .venv + installs deps from pyproject.toml, writes uv.lock
```

uv manages the Python interpreter and the virtualenv — no manual `venv` or
`activate`. `uv run` also auto-syncs, so the explicit `uv sync` is optional.

## Run

In one terminal, start the server (from the repo root):

```bash
scripts/run-server.sh
```

In another:

```bash
cd agent

# 1. Sanity check — two random bots play each other (watch at localhost:8000)
uv run python smoke_test.py

# 2. Play against it yourself
uv run python play.py

# 3. Phase 1 yardstick — does max-damage beat random? (server running)
uv run python evaluate.py            # ~100 games, prints win rate

# 4. Phase 2 — train the RL agent (server running; first `uv sync` installs torch)
uv run python train.py --smoke       # tiny run, just checks the loop works
uv run python train.py               # real run; checkpoints to models/

# tests / lint (engine-free core; no server needed)
uv run pytest
uv run ruff check
```

To play it: open http://localhost:8000, pick any name, build or **import a
[Gen 1] OU team** (you can paste `teams/gen1ou-sample.txt`), then find the user
`CinnabarBot` and challenge it to a `[Gen 1] OU` battle. It plays random legal
moves.

## Status

Phase 2: a custom PyTorch RL agent (`cinnabar/rl/`) trained by REINFORCE-with-
baseline on a sparse win/loss reward, plus `RandomPolicy` and `MaxDamagePolicy`
baselines and `evaluate.py`. The RL agent is just another `Policy`, so it plays,
evaluates, and (soon) battles humans through the same path. See `../docs/roadmap.md`.
