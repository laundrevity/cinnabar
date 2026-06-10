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
│   ├── include/cinnabar/engine.hpp + encoder.hpp + gen1_data.hpp (GENERATED)
│   ├── src/engine.cpp      # battle mechanics | src/encoder.cpp: C++ observation encoder +
│   │                       #   heuristic pilots (bit-parity with the Python featurization; see below)
│   ├── bindings/bind.cpp   # pybind11 module: import cinnabar_engine
│   └── tools/              # gen_data.py (codegen) | ref_trace.js + trace_diff.py (fidelity harness)
├── server/             # Pokémon Showdown lives here
│   └── pokemon-showdown/   # git submodule (added by scripts/setup.sh); also the fidelity oracle
├── agent/              # Python agent (uv project)
│   ├── pyproject.toml      # deps (poke-env) + dev group (pytest, ruff)
│   ├── cinnabar/           # state.py + policy.py + encoding.py + rl/ (engine-free core)
│   │                       #   showdown.py (poke-env adapter) | engine_cpp.py (C++ engine adapter)
│   ├── play.py             # Phase 0: accept human challenges in Gen 1 OU
│   ├── play_cli.py         # play the agent in the TERMINAL (C++ engine, no server/browser)
│   ├── train.py            # PPO training via Showdown (poke-env)
│   ├── train_engine.py     # PPO training in-process on the C++ engine (vectorized); --clauses for OU Sleep/Freeze Clause
│   ├── evolve_teams.py     # team optimizer (team construction): evolve teams vs a meta anchor, agent/heuristic pilots
│   ├── cotrain.py          # co-training driver: alternate evolve-teams <-> train-agent, growing team archive, no priors
│   ├── search.py           # decision-time search (1-ply lookahead on the engine, value-head leaf); +19% over raw
│   ├── expert_iter.py      # expert iteration: distil search into the policy (AlphaZero-lite), value head -> win/loss
│   ├── diagnose_policy.py  # quantify pilot weaknesses (switch-rate, re-sleep, vs a StallerPolicy)
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
- **Play in the terminal (no server):** `cd agent && uv run python play_cli.py --ckpt models_fast/pg_best.pt` — interactive battle on the C++ engine, you vs the agent (pilots: `search` (default) / `raw` / `heuristic` / `staller`; search takes the usual `--top-k/--minimax/--opp-top-k/--opp-temp/--value-ckpt` knobs). Partial information both ways; clauses on by default (`--no-clauses`); `--show-eval` prints the agent's win-prob each turn. In battle: number = action, `t` = team, `q` = forfeit.
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
- **C++ observation encoder (the throughput path, June 2026):** profiling showed the engine at
  ~250k turns/s but Python `build_state`+`featurize` eating ~80% of rollout time, so the
  featurization now runs in C++ (`engine/src/encoder.cpp`): `ce.encode_batch` returns padded
  numpy feature tensors, `ce.step_pair` does reveal bookkeeping + step, `ce.select_heuristic`
  is the C++ twin of MaxDamage/SmartHeuristic/Staller. **Static data is NOT re-derived in
  C++** — `engine_cpp.register_encoder(static)` uploads the same poke-env tables build_state
  uses, and `tests/test_encoder_parity.py` asserts **bit-identical float32 features** and
  decision-identical heuristic picks across whole battles (clauses, partial info, generated
  movesets, dup-move edge cases). Net effect: rollouts/evals ~12x (→ ~2,200 battles/s),
  training ~4.5x end-to-end (~450 battles/s; the PPO update's backward is now the floor).
  `train_engine._run` dispatches to the fast path; `--frame-stack > 1` falls back to the
  Python loop (`_run_py`). The per-action net got a factored first layer
  (`ActionScorer._layer1`: global half computed once per state, not per action) — same
  parameters/checkpoints, ~2x faster PPO updates. If you change `encoding.py`, change
  `encoder.cpp` to match and rebuild — the parity test (and a GLOBAL_DIM check in
  `register_encoder`) will catch drift. **Search runs on the same fast path** (~5x):
  `search_action_values`/`_minimax` push the whole leaf loop into one `ce.search_leaves`
  call + one batched value forward whenever the leaf is the value head or a heuristic
  playout (`tests/test_search_fast.py` pins fast == slow per decision); a net opponent-model
  (expert_iter `--opp-model policy`) still gets fast leaves. Only a NET-piloted *rollout
  leaf* falls back to the per-leaf Python loop.
- **Ladder rates search pilots (the browser configuration):** `ladder.py --search-ckpts
  ckpt.pt` adds a SearchPilot player (`--search-rollouts/-top-k/-minimax/-opp-top-k/-opp-temp`),
  and a `staller` baseline is included by default (`--no-staller` to drop) — raw policy,
  search-piloted policy, and the heuristics all rank on one Elo scale, so "what is lookahead
  worth" is a ladder read, not a one-opponent win%.
- **Train with OU clauses:** add `--clauses` (Sleep + Freeze Clause — a 2nd foe-inflicted sleep/freeze fails). Modeled as a per-`Side` flag, **default off** so the bit-for-bit harness (clause-free `gen1customgame`) is untouched; `ladder.py --clauses` rates under the same rule. Without it the agent over-values sleep (it can sleep your whole team in training). GLOBAL_DIM gained 2 clause-perception features — warm-start old nets with `pad_checkpoint.py`.
- **Evolve teams (team construction):** `cd agent && uv run python evolve_teams.py --ckpt models_clauses/pg_best.pt --pilots net,heuristic --pop 24 --gens 30 --clauses --out evolved` — fitness = win-rate vs a fixed anchor (`--anchor-dir`, default `teams/`) under a pilot panel; top teams written as Showdown `.txt`. Single-pass evolution coadapts to the pilot's blind spots (a weak pilot can't value Snorlax / punish over-statusing), so for real team discovery use co-training instead. Pilot ckpt must match the current GLOBAL_DIM (loaders auto-pad older ones).
- **Co-train agent + teams (`cotrain.py`):** `cd agent && uv run python cotrain.py --init models_clauses/pg_best.pt --rounds 6 --clauses --out cotrain` — each round evolves teams vs the current agent (scored against a GROWING archive seeded from random teams, no human priors), then retrains the agent on the whole archive. Strategy emerges from the loop. Watch the per-round ladder margin-over-smart for collapse; `--dry-run` prints the commands. The emergence-pure answer to team construction (vs hand-coded constraints or hand-picked anchors).
- **Decision-time search (engine path only):** `cd agent && uv run python search_eval.py --ckpt models_clauses/pg_best.pt --battles 100 --rollouts 3 --clauses` — 1-ply lookahead (clone+reseed the engine, value-head leaf, heuristic opponent model) played ~**+19%** over the raw greedy policy vs the staller *for that checkpoint*. **The lift is checkpoint-dependent, re-measure per net** (2026-06-10: on the fresh `models_fast/pg_best` league net, the same search configs measured **-8 to -11% vs the staller** even with a freshly calibrated value leaf, while the ladder still rated `+search[k3]` ~+27 Elo over raw against the full field — search helped broadly but hurt against the staller specifically). The lever past the self-play ceiling: the agent computes a better move than its policy without a stronger opponent. `--sweep` maps rollouts/leaf/top-k headroom. **`--top-k N` = policy-prior gating** (search only the policy's top-N actions; plumbed through `search_eval.py` / `probe_eggy.py` / `evolve_teams.py --top-k` / both `*_search_battle` fns): without it value-leaf search *bypasses* the policy and un-learns its discipline — measured ~20–46% clause-wasted sleep clicks in search-mirror play that the raw policy never makes. The policy proposes, the value head disposes; also faster (fewer clones). **Diagnose first** (`diagnose_policy.py`): all the raw-pilot weaknesses I hypothesized (switch-loop, can't-play-defense) were refuted by measurement — the net is a competent *average* player; its holes only a strong human triggers.
- **Expert iteration only works when the teacher is stronger (2026-06-10):** a 6-round v3 run
  (`--margin 0.05`, KL anchor, monotone gate) from `models_fast/pg_best` produced **zero
  accepted rounds** — every round's distilled net lost to the init on the two-opponent eval and
  was reverted. Cause: for that checkpoint, gated search is NOT stronger than the raw policy
  (see the search-lift caveat above), so the "teacher" had nothing to teach. Check
  `search_eval` shows a positive lift BEFORE spending an expert-iter run; until search beats
  raw again, improve the policy directly (training mix/breadth) or the value function.
- **Staller anchor + two-opponent gating (`train_engine --anchor staller|mix`):** self-play +
  smart-anchor training never shows the patient para+recover style, which is exactly how
  humans beat the agent (browser + play_cli ground truth). `--anchor mix` rotates
  smart/staller/maxdamage on anchor iters, evals print a staller line, and `pg_best.pt` is
  gated on the MEAN of smart+staller (smart-only selection bled to stallers).
- **Expert iteration (`expert_iter.py`, v2):** `cd agent && uv run python expert_iter.py --init models_wf/pg_best.pt --rounds 5 --games 60 --gen-teams --clauses --out ei2` — self-play where both sides move by **policy-prior gated search** (`--top-k 3`), distil the **soft search distribution** (softmax over candidate Q-values, `--tau`) + outcome (value MSE) back into the net under a **KL trust-region anchor to the frozen init** (`--anchor-coef`), with the lookahead modelling the opponent as the **current policy** (`--opp-model`). v1 (hard argmax CE, heuristic opp-model, no anchor) *degraded* the policy — those three causes are exactly what v2 fixes; `--anchor-coef 0` reproduces the v1 failure, don't. `pg_best.pt` is only written when a round beats the running best on the two-opponent eval (smart + staller). Watch BOTH eval lines per round — the v1 collapse showed first vs the staller.

## Conventions

- Showdown is a **submodule pinned to a commit** — bump it deliberately, not implicitly.
- `--no-security` is for **local use only**; never expose such a server publicly.
- Trained weights, replays, and logs are git-ignored; keep large artifacts out of the repo.
