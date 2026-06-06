# agent/

The AI agent that plays battles. **Language not yet decided** — placeholder for now.

The leading option is **Python** (`poke-env` ships a `RandomPlayer` for Phase 0 and a
Gymnasium wrapper for RL; the whole ML ecosystem — PyTorch, stable-baselines3 — lives here).
TypeScript is the alternative if a single-language stack with Showdown is preferred, at the
cost of a thinner ML ecosystem.

Phase 0 target: connect to the local server, accept a Gen 1 OU challenge, and pick a random
legal move each turn. See `docs/roadmap.md`.
