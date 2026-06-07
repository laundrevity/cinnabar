# Scaling self-play: @pkmn/engine integration plan

Goal: replace Showdown (our rollout bottleneck) with a ~1000× faster in-process
engine so we can run the orders-of-magnitude more self-play that nuanced play needs.
This is a research-scale effort, not a tweak. This doc scopes it honestly.

## Engine status — read this first

`@pkmn/engine` is **pre-v0.1 and explicitly not ready to depend on yet.** Its README:

> This project is under heavy development and currently the `main` branch contains
> numerous breaking changes which may not work and which are not fully documented.
> Please wait for the forthcoming initial v0.1 release before depending on this project.

Gen 1/2 ("Stage 1") is *in progress*. So the upside (it's exactly our generation, and
~1000×) is real, but we'd be building on a moving, partially-documented target. The PoC
below exists to find out whether it's usable *today* before we sink weeks in.

## How a Python loop can drive a Zig engine

The engine is a low-level library, not a simulator (no team validation, no format rules,
full-information). Interface options:

| Path | What it is | For us |
| --- | --- | --- |
| **`libpkmn` C API** (`pkmn.h`) | Zig-built shared lib; `pkmn_battle_update(...)` + `choices`. Opaque battle struct — you decode it yourself using the JSON size/offset dumps in `src/data`. | Full control, most work. ctypes/cffi from Python. |
| **JS/TS `@pkmn/engine`** | Ergonomic Node/WASM driver. `npm install` auto-builds the native addon. | Easiest to *validate* the engine works; wrong language for our loop. |
| **Zig `pkmn`** | Native package. | Would mean rewriting training in Zig. No. |
| **PyKMN** (community) | A third-party **Python binding to libpkmn** (MIT, github.com/AnnikaCodes/PyKMN). | The most pragmatic Python on-ramp — *if* it's current. README warns external bindings "may not be up-to-date/complete/correct." |

Realistic Python options, best case first: **(1) PyKMN if it tracks the current engine;
(2) our own ctypes bindings against `libpkmn` using the JSON struct offsets; (3) a Node
worker hosting `@pkmn/engine` that Python drives over fast local IPC in big batches.**

## What changes in our codebase (and what doesn't)

The engine-free core is the payoff here: `state.py`, `policy.py`, `encoding.py`, and
`cinnabar/rl/` **stay**. We write a **new adapter** (the `@pkmn/engine` analogue of
`showdown.py`) that:

1. initializes a battle from one of our teams (encode team → engine's binary battle struct),
2. asks the engine for legal `choices` and maps them to our `Action`s,
3. steps with `update(c1, c2)` and decodes the binary battle state → our `BattleState`,
4. **re-implements fog-of-war ourselves** — the engine is full-information, so to keep our
   partial-observability (revealed opponent team) we mask state in the adapter.

It's a bigger adapter than `showdown.py`, but the architecture decision from day one
("retarget by rewriting one file") is exactly what makes this tractable.

Then: a new **vectorized self-play loop** (synchronous, in-process, thousands of battles —
no asyncio/WebSocket), and only *now* does GPU/Metal training matter (big batches of cheap
rollouts → the network update stops being a rounding error).

## Important: scale alone may not be the whole answer

The current state-of-the-art for *human-level* Pokémon — "Human-Level Competitive Pokémon
via Scalable Offline Reinforcement Learning with Transformers" (2025) — got there with a
**transformer** policy and **offline RL on a large dataset of human ladder replays**, not
pure fast self-play. Takeaway: the engine buys us **scale**, but human-level also needed an
**architecture with memory** (transformer) and, notably, **human replay data**. Expect to
need the architecture lever too; the engine is necessary-not-sufficient.

## Staged plan

- **Stage 0 — Proof of concept (de-risk before committing).** On the dev machine: build/run
  the engine at all, and benchmark Gen 1 throughput. If it won't build or PyKMN is stale,
  we learn that for the cost of an afternoon, not weeks.
- **Stage 1 — Python binding.** Get a single Gen 1 battle stepping from Python (PyKMN, else
  ctypes against `libpkmn`). Benchmark battles/sec vs our current Showdown rate.
- **Stage 2 — New adapter.** Engine battle ↔ our `BattleState`/`Action`, with fog-of-war.
  Validate it against `showdown.py` (same policy should behave the same).
- **Stage 3 — Vectorized self-play + GPU.** Batch thousands of battles; move the (now
  worth-it) network to MPS/GPU. Re-run league + dense-reward training at 100×+ the battles.
- **Stage 4 (likely needed) — architecture.** Transformer/recurrent policy for planning,
  per the SOTA finding.

## Proof of concept (run on the dev machine)

```sh
# A) Does the engine build & run here at all? (validates Zig toolchain + throughput, in JS)
mkdir -p /tmp/pkmn-poc && cd /tmp/pkmn-poc && npm init -y
npm install @pkmn/engine @pkmn/dex @pkmn/data    # postinstall fetches zig + builds the addon
#   then run the README's JS example, looping N random Gen 1 battles, and time it.

# B) The real question — can Python drive it?
#   Try PyKMN (github.com/AnnikaCodes/PyKMN): install, run one Gen 1 battle, benchmark.
#   If PyKMN is stale/broken against the current engine, fall back to ctypes vs libpkmn:
curl -L https://github.com/pkmn/engine/archive/refs/heads/main.zip -o engine.zip
unzip engine.zip && cd engine-main
zig build --prefix ./out -Doptimize=ReleaseFast -Dshowdown -Dlog   # builds libpkmn-showdown
#   -> then a minimal ctypes script calling pkmn_battle_update in a loop; benchmark.
```

Report back: (1) does it build, (2) does Python drive it, (3) battles/sec vs Showdown.
That decides whether the engine path is real *now* or blocked on v0.1.
