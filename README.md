# Cinnabar

Building and training an AI to play **Pokémon battles**, using
[Pokémon Showdown](https://github.com/smogon/pokemon-showdown) as the game engine.

The plan is incremental: start with a bot that plays random legal moves (which you can beat
in the browser), then make it progressively smarter — heuristic baseline, then reinforcement
learning and self-play.

**Current scope:** Gen 1 OU — chosen for its mechanical simplicity (no abilities, no held
items, small movepools) while still being a real competitive format.

## Quickstart

```bash
# 1. Pull in Pokémon Showdown (as a git submodule) and build it
scripts/setup.sh

# 2. Start a local server (no logins / rate limits — for local use only)
scripts/run-server.sh
# -> http://localhost:8000   WebSocket: ws://localhost:8000/showdown/websocket
```

Open the printed URL in a browser to play. Your agent will connect over the WebSocket
endpoint above.

> The submodule fetch and `npm install` need network access to GitHub and npm, so run
> `setup.sh` on your own machine.

## Roadmap (short version)

| Phase | Goal |
|-------|------|
| **0** | Random-move bot, playable in the browser. Proves the full pipeline end-to-end. |
| **1** | Simple heuristic baseline (e.g. max-damage) to beat the random bot — a yardstick. |
| **2** | Reinforcement-learning agent vs. the baselines; iterate on state/reward design. |
| **3** | Self-play at scale; swap in a faster engine *only if* throughput is proven to be the wall. |

See [`docs/roadmap.md`](docs/roadmap.md) for the detailed plan and the reasoning behind the
key architecture choices.

## Layout

- `server/` — Pokémon Showdown (git submodule) + run scripts
- `agent/` — the AI agent (language TBD)
- `teams/` — Gen 1 OU teams in Showdown export format
- `docs/` — roadmap and design notes
- `scripts/` — setup and server helpers

See [`CLAUDE.md`](CLAUDE.md) for the project context and decisions in full.
