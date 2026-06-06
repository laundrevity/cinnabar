# Roadmap

The guiding principle: **prove each layer works before optimizing the one beneath it.**
Get a learning loop that converges on the real game first; make it fast later, only if you
must.

## Phase 0 — Random bot, playable in the browser

Goal: wire the whole pipeline together end-to-end with the dumbest possible policy.

- Run a local Showdown server (`scripts/run-server.sh`).
- Stand up an agent that connects, accepts challenges, and picks a **random legal move**
  each turn (and a random switch when forced).
- Play it yourself in the browser to confirm the loop works.

Why first: this surfaces every integration problem (server, protocol, team format, move
legality, forced switches, timeouts) while the policy is trivial. `poke-env` ships a
`RandomPlayer`, so most of this is connection plumbing rather than logic.

**Done when:** you can challenge the bot to a Gen 1 OU battle and play a game to completion.

## Phase 1 — Heuristic baseline

Goal: a non-learning opponent that's clearly better than random, to serve as a yardstick.

- Implement a **max-damage** policy (pick the move with the highest expected damage given
  type effectiveness), with simple switch logic.
- Optionally a slightly smarter rules bot (respect status, avoid switching into bad matchups).

Why: you can't tell if learning is working without a fixed reference point. "Beats random"
and "beats max-damage" are your first two real milestones.

**Done when:** the heuristic reliably beats the random bot, and you have a harness that plays
N games between any two agents and reports win rate.

## Phase 2 — Reinforcement learning

Goal: an agent that learns to beat the baselines.

- Decide the **agent language** (Python is the default — Gymnasium + PyTorch + `poke-env`'s
  Gym wrapper).
- Design the hard parts:
  - **State representation** — encode active + benched Pokémon, HP, status, boosts, known
    moves, hazards. Start minimal.
  - **Action space** — moves + switches, with legal-action masking.
  - **Reward** — sparse win/loss to start; consider shaping (damage dealt/taken, faints)
    carefully, since shaping can teach the wrong thing.
- Train against the random and heuristic baselines first (a stationary opponent is easier to
  learn against than a moving one).

**Done when:** the RL agent beats the heuristic baseline by a clear margin.

## Phase 3 — Self-play and scale

Goal: push past the baselines via self-play, and only now worry about speed.

- Self-play / league play (train against past versions to avoid overfitting one opponent).
- **Profile before optimizing.** If, and only if, simulation throughput is the proven
  bottleneck, swap Showdown's sim for [`@pkmn/engine`](https://github.com/pkmn/engine)
  (Zig, ~1000× faster, built for ML; covers Gen 1/2, which matches our scope).
- Evaluate against external bots or the ladder if you want a public benchmark.

## Open questions to resolve as you go

- Agent language (Python vs TypeScript) — lock before Phase 2.
- Team strategy: fixed team(s) for training, with team-building treated as a separate
  problem layered on later.
- Algorithm: start simple (DQN / PPO) before anything exotic.

## Decisions already made

- Game engine: **Pokémon Showdown** (submodule), not a custom reimplementation.
- Format: **Gen 1 OU** (mechanical simplicity).
- No custom C++ simulator. Speed, if ever needed, comes from parallelism then `@pkmn/engine`.
  See `CLAUDE.md` for the full reasoning.
