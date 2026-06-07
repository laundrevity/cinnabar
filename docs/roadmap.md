# Roadmap

The guiding principle: **prove each layer works before optimizing the one beneath it.** Get a
learning loop that converges on the real game first; make it fast later.

This is the plan as it actually unfolded — including a deliberate mid-project reversal (see the
"Decisions" section): the agent was first built against Showdown via `poke-env`, then a custom
C++ engine was built, validated bit-for-bit against Showdown, and the agent moved onto it.

## Phase 0 — Random bot, playable in the browser ✅

Wire the whole pipeline together with a random-legal-move policy: a `poke-env` player that
connects to a local Showdown server, accepts challenges, and plays Gen 1 OU to completion.
Surfaces every integration problem (protocol, team format, forced switches) while the policy
is trivial.

## Phase 1 — Heuristic baseline ✅

A non-learning yardstick: a type-aware **max-damage** policy (`base_power × type_multiplier ×
STAB`, with switch fallback), plus an N-game evaluation harness. "Beats random" and "beats
max-damage" become the first real milestones.

## Phase 2 — Reinforcement learning ✅

Python (Gymnasium-free; PyTorch + `poke-env`). A **per-action scorer**: the net scores each
legal action from `[global features ++ action features]` and softmaxes, which handles a
variable action count and makes illegal-action masking automatic. Engine-free `state.py` /
`encoding.py` / `policy.py` behind an adapter; REINFORCE-with-baseline → **clipped PPO**;
sparse → shaped → dense reward options. Beats the baselines.

## Phase 3 — Self-play and scale ✅ (and the engine reversal)

Self-play / league play against past snapshots. Then the key decision: rather than chase raw
throughput inside Showdown, **build a custom Gen 1 engine** (see below) and train on it.

## Engine phases (`engine/`) — the reversal ✅

1. **v0** — damage/stat formulas, type chart, 1v1 loop.
2. **Full battles** — 6v6, switching, status + moves, healing, Reflect, Explosion.
3. **Fidelity** — generate static data from Showdown (`gen_data.py`); build the **exact
   turn-for-turn differential harness** (`trace_diff.py`) and close every divergence: Gen 5
   LCG + call-sequence, stat rollover, dual-type rounding, status RNG, stat stages, PP/Struggle,
   switching. Validated bit-for-bit over tens of thousands of battles.
4. **Python bindings + adapter** — pybind11 module + `agent/cinnabar/engine_cpp.py`, so the
   engine-free RL core runs on the engine unchanged.
5. **Scale** — vectorized in-process self-play (batched rollouts + batched PPO update),
   ~500 battles/sec; the agent trains from random to beating max-damage in minutes.

## Where it stands / open threads

The build is done: a faithful, validated engine and an RL agent that trains on it fast.
What remains is open-ended ML iteration, roughly in value order:

- **Self-play / league curriculum** to push past the max-damage plateau toward strong play.
- **Breadth**: team variety (parse `teams/`), wider modeled-move coverage (Counter, Substitute,
  multi-hit, …), reward/hyperparameter tuning.
- **Throughput**: move `build_state`/featurization into C++ if more speed is wanted.

## Decisions

- **Format: Gen 1 OU** — mechanical simplicity.
- **Custom C++ engine (a reversal of the original "no custom sim").** The project began on
  Showdown via `poke-env` (the right call for a fast learning loop). The engine was then built
  as a deliberate engineering project — kept honest by **differential testing against Showdown**
  (the fidelity oracle), not by reimplementing from memory. Static game data is generated from
  Showdown, never hand-typed.
- **poke-env / Showdown kept behind an adapter**, so the decision core is engine-agnostic and
  the same agent runs on either backend.
- **Algorithm:** per-action PPO; self-play/league for the curriculum.
