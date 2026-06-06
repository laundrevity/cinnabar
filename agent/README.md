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

# 3. Compare policies — baselines or a trained checkpoint (server running)
uv run python evaluate.py                                              # maxdamage vs random
uv run python evaluate.py --a pg --b maxdamage --checkpoint models/pg_best.pt -n 500

# 4. Phase 2 — train the RL agent (PPO; server running; first `uv sync` installs torch)
uv run python train.py --smoke                                            # tiny run, checks the loop
uv run python train.py --opponent random --step-penalty 0.01 --iters 200  # warm up vs random
uv run python train.py --opponent maxdamage --init models/pg_best.pt \
    --step-penalty 0.01 --iters 200 --out models_md                       # then push vs the baseline
uv run python train.py --opponent self --init models_md/pg_best.pt \
    --step-penalty 0.01 --iters 300 --out models_sp                       # Phase 3: self-play

# tests / lint (engine-free core; no server needed)
uv run pytest
uv run ruff check
```

To play it: open http://localhost:8000, pick any name, build or **import a
[Gen 1] OU team** (you can paste `teams/gen1ou-sample.txt`), then find the user
`CinnabarBot` and challenge it to a `[Gen 1] OU` battle. It plays random legal
moves.

## Status

Phase 3 (in progress): a custom PyTorch RL agent (`cinnabar/rl/`) trained by PPO
(REINFORCE selectable via `--algo`) on a sparse win/loss reward, with per-switch and
team-state observations. It beats the `MaxDamagePolicy` baseline; now training by
**self-play** (`--opponent self`) against evolving snapshots of itself, measured
against the random / max-damage yardsticks. The RL agent is just another `Policy`.
See `../docs/roadmap.md`.
