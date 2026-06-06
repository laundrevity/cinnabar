# agent/

The AI agent that plays battles. **Language: Python** (via
[`poke-env`](https://poke-env.readthedocs.io/), which handles the Showdown
WebSocket protocol and battle-state tracking).

## Design: poke-env behind an adapter

The package is split so the decision-making core never touches poke-env:

```
cinnabar/
  state.py     # our BattleState + Action types        (no poke-env)
  policy.py    # Policy ABC + RandomPolicy              (no poke-env)
  showdown.py  # PolicyPlayer: the poke-env adapter     (the only poke-env code)
play.py        # Phase 0: accept human challenges in Gen 1 OU
smoke_test.py  # bot-vs-bot sanity check, watchable in the browser
```

Every future agent (heuristic, RL, self-play) implements `Policy.select_action`.
If we ever swap poke-env for a faster engine, only `showdown.py` changes.

## Setup

Requires Python 3.10+ and a local Showdown server (`../scripts/run-server.sh`).

```bash
cd agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

In one terminal, start the server (from the repo root):

```bash
scripts/run-server.sh
```

In another, with the venv active:

```bash
cd agent

# 1. Sanity check — two random bots play each other (watch at localhost:8000)
python smoke_test.py

# 2. Play against it yourself
python play.py
```

To play it: open http://localhost:8000, pick any name, build or **import a
[Gen 1] OU team** (you can paste `teams/gen1ou-sample.txt`), then find the user
`CinnabarBot` and challenge it to a `[Gen 1] OU` battle. It plays random legal
moves.

## Status

Phase 0 (random policy). Next: a heuristic `MaxDamagePolicy` baseline, then RL.
See `../docs/roadmap.md`.
