# HANDOFF.md

Continuity note for a fresh session. Read `CLAUDE.md` first for what the project is, the repo
layout, and build/run commands — this file is **what happened in the last session, what we learned,
and where the open frontier is.** It does not repeat CLAUDE.md.

---

## TL;DR — where things stand

- **Strongest agent:** `agent/models_clauses/pg_best.pt` (Gen 1 OU, clause-trained). On a clauses
  ladder it's **+63 over smart** vs genteams3's +36 (i.e. **+27 Elo** from clause training).
- **Decision-time search is the real lever.** 1-ply lookahead on the engine (`agent/search.py`)
  plays **+19% over the raw policy vs the staller and +27% vs smart.** Validated, robust. The raw
  greedy policy leaves a lot on the table; search recovers it by computing instead of recalling.
- **Team construction is substantially unblocked — by fixing the JUDGE, not the search.** Using the
  search net as the evolution pilot (`evolve_teams.py --pilots search`) brought Snorlax, Chansey and
  a sleeper back onto every top team (the raw-policy judge had cut them and produced Persian/Hypno
  junk). The central thesis of the session — *the judge was the bottleneck* — is confirmed.
- **Two things are NOT solved.** (1) Expert iteration (distilling search into the policy to make it
  free) **failed** — it degraded the policy across the board. (2) The agent still under-values
  **Exeggutor** specifically (it picks Jynx/Gengar as the sleeper). Both point at the same wall:
  **strategic / positional depth.** The agent is tactically competent but strategically shallow.

**The single most important lesson from this session: measure before fixing.** Five confident
hypotheses about the agent's weaknesses were *refuted* by a quick diagnostic before any fix was
built. The diagnostics were the hero; the predictions were the goat. Keep that discipline.

---

## The two problems we were chasing

The session started from a browser loss (Conor beats the agent easily) that exposed two issues:

1. **The agent is dumb about status / Sleep Clause** — it spammed Lovely Kiss into a clause-locked
   foe, wasting turns.
2. **Team construction was unaddressed** — the agent is handed teams; it should build its own, and
   "all teams in the pool" is a crude way to do it.

Both got worked. Status → clauses (below). Team construction → evolution / co-training / a search
judge (below).

---

## What was built this session (with commit refs)

- **OU Sleep + Freeze Clause** as a per-`Side` engine flag, **default off** so the bit-for-bit
  fidelity harness (clause-free `gen1customgame`) is untouched; training/ladder opt in with
  `--clauses`. + a clause *perception* feature (GLOBAL_DIM 166→168). Commit `acecbb8`. Result:
  `models_clauses` = +27 Elo over the clause-blind net under clauses. See memory `cinnabar-clauses`.
- **Auto-pad on load** (`d6600e1`, extended `16e5869`): older/smaller checkpoints auto-pad to the
  current dims in `ladder.py` / `play.py` / `train_engine --init`. No manual `pad_checkpoint.py` step.
- **`evolve_teams.py`** — team optimizer. Started coevolutionary (`ddde8c6`), reworked to a fixed
  meta-anchor + pilot panel (`f1b6b9a`), and given a **`search` pilot** (`99bc44e`). A one-sleeper
  cap was added then **reverted** (`ddde998`) — see "Conor's principles".
- **`cotrain.py`** (`4c5a312`) — co-training driver (evolve teams ↔ retrain agent, growing archive,
  no priors). Ran once; produced a fast-frail-attacker meta (Eggy/Chansey cut) → exposed the judge as
  the bound.
- **`diagnose_policy.py`** (`49b4fcd`, `8c03861`) — quantify pilot weaknesses: switch-rate, re-sleep
  rate, game length, and an A/B piloting the same defensive team vs the heuristic. Plus
  **`StallerPolicy`** (`74c574f`) — a patient paralysis+recovery opponent.
- **Decision-time search** (`ba71724`): `Battle.clone()` / `Battle.reseed()` bindings + `search.py`
  (1-ply lookahead, value-head leaf, heuristic opponent model, fresh-dice rollouts) + `search_eval.py`
  (paired search-vs-raw). Rollout-to-terminal leaf + sweep + re-sleep tracking added (`23f4bb8`).
- **Clause-fail action feature** (`16e5869`): a direct per-action "this sleep/freeze will fail right
  now" signal (ACTION_DIM 22→23). **Needs a retrain to take effect** (zero-padded into the current
  net). `pad_checkpoint` now handles action-dim growth too.
- **`expert_iter.py`** (`fd0f049`) — AlphaZero-lite distillation of search into the policy. **FAILED**
  (see below).

---

## The big lessons (read this before proposing fixes)

### Measure first — five refuted hypotheses

Each of these was a confident diagnosis that a quick measurement killed. **Do not skip the
measurement step.**

1. *"The agent switch-loops pathologically."* → measured **12.3%** voluntary switch rate (heuristic:
   0.1%). Not pathological. The browser ping-pong was a rare, strong-human-triggered tail, not the
   average.
2. *"It can't pilot defense."* → it pilots the *same* defensive team **better** than the heuristic
   (64.7% vs 50%). Refuted.
3. *"It folds to patient stall."* → **58.7%** vs the staller (vs 64.2% vs the attacker). A modest
   drop, not a fold. My hand-built staller isn't as strong as Conor.
4. *"Search will fix the re-sleep for free."* → search re-sleeps at ~15–27%, ≈ the raw rate. It does
   **not** subsume the clause bug; hence the explicit clause-fail action feature.
5. *"Expert iteration will compound search into a stronger policy."* → it **degraded** the policy
   (62.9% → 40% vs smart, 27% vs staller) despite search being a genuinely better target (+27% vs
   smart). Method failure, not premise failure.

The recurring shape: **the agent is a competent *average* player whose real holes only a strong human
triggers — and you cannot reproduce a strong human with a hand-built heuristic, because building one
is the hard problem itself.** That is why heuristic-based diagnostics keep saying "the agent is fine"
while Conor beats it. The honest implication is that progress needs *strength*, not more probes.

### Conor's principles (he will hold you to these)

- **Emergence, not hand-coding.** He rejected a one-sleeper-per-team cap outright: two sleepers is a
  legitimate strategy, and hard constraints amputate the strategy space before the search explores
  it. Don't encode meta priors as constraints — let strategy emerge (search, co-training). Symptoms
  like over-statusing or a dead Snorlax are signs of a *weak judge*, not missing rules. See memory
  `cinnabar-emergence-principle`.
- **No hand-fed teams.** Team discovery must be organic (random seeds + the loop), not seeded from
  teams he hand-picks. A *policy* warm-start is fine; a *team* prior is not.
- **Fidelity is sacred.** The engine is validated bit-for-bit vs Showdown. Anything that could change
  that (e.g. clauses) is a default-off flag. Always confirm `trace_diff.py sweep` stays N/N and
  `validate_moves.py` 24/24 after engine changes.
- **He is a strong RBY OU player.** His reads ("Eggy is ubiquitous", "that lead is bad", "this team
  is viable") are reliable ground truth — weight them heavily.

---

## Why expert iteration failed (so it isn't blindly retried)

`expert_iter.py` distils the **argmax** of a 1-ply value-leaf search via hard cross-entropy, on
both-sides-search self-play, with **no replay/anchor** to the original policy. It tanked the policy.
The premise is fine (search is +27% better), so it's a method problem. Three likely causes, all
fixable but none a one-liner:

1. **Opponent-model mismatch in self-play:** each side's search assumes a *heuristic* opponent while
   actually facing *search* → mis-calibrated targets.
2. **No replay / anchoring:** pure imitation on a narrow self-play distribution overwrote a coherent
   PPO policy (catastrophic forgetting).
3. **Hard CE to argmax, not a distribution,** of a target the net can't fully represent (CE plateaus
   ~1.3) → an incoherent greedy policy.

Real AlphaZero avoids all three (policy-prior search, distributional/visit-count targets, replay). If
revisited, fix those — don't just rerun. **`agent/ei/pg_best.pt` is the broken net; do not use it.**

---

## Current state of the model/team files (`agent/`)

- `models_clauses/pg_best.pt` — **the strong net.** Use as the search judge / browser agent / warm
  start. (ACTION_DIM 22; auto-pads to 23 on load.)
- `models_genteams3/pg_best.pt`, `models_genteams2/...` — older, clause-blind. Superseded.
- `ei/pg_best.pt` — **broken** (failed expert iteration). Ignore.
- `evolved_search/` — the search-judged teams (Snorlax/sleeper present; #1 is viable). The good output.
- `evolved/`, `cotrain/` — earlier, weaker-judge outputs (off-meta).
- `probe_teams/strong.txt` — a hand-built strong team kept OUT of `teams/`, for isolating pilot
  quality from team quality in the browser (`play.py --teams-dir probe_teams`).

---

## Open questions & recommended next steps

**The frontier is strategic depth.** The team-construction judge problem is largely solved by search;
what remains is that the agent (raw *and* 1-ply search) is strategically shallow.

1. **Why is the agent bad at Exeggutor?** (Conor's open question.) Hypothesis (un-measured): Eggy's
   value is *positional* — bulk/repeated switch-ins, sleep-target selection, Explosion *timing* —
   multi-turn value that 1-ply search can't see, whereas Jynx/Gengar deliver immediate "click the
   nuke" value the agent does capture. **Recommended probe:** pilot one strong team with Eggy in the
   sleeper slot vs the same team with Jynx swapped in, same opponent; compare win-rate AND log Eggy's
   behaviour (does it ever Explode? does it waste the sleep?). If it wins with Jynx but flounders with
   Eggy, the hypothesis holds — it's a piloting-depth gap. (Build it; measure before concluding.)
2. **Deeper search.** 1-ply value-leaf is the practical config (rollout-to-terminal leaf needs too
   many samples to denoise). A real 2-ply / shallow MCTS that uses the *policy as a prior* might
   capture positional value — and would also be the right substrate for a corrected expert iteration.
3. **Fix expert iteration properly** (replay/anchor + matched opponent model + distributional
   targets) to make the +27% free at inference. Only worth it if you accept it's a research iteration.
4. **Browser ground truth.** Reconstruct engine state from poke-env so the *search* agent plays Conor
   directly — the honest test of "is it actually good now." A bounded build (partial-info recon).
5. **The clause-fail feature is added but untrained.** The next training run (a corrected expert
   iteration, or any retrain from `models_clauses`) will activate it; re-run `diagnose_policy` after
   to confirm the re-sleep collapses.

---

## Key commands & gotchas (session-specific; see CLAUDE.md for the rest)

- **Diagnose the pilot:** `uv run python diagnose_policy.py --ckpt models_clauses/pg_best.pt --battles 300 --clauses`
- **Measure search:** `uv run python search_eval.py --ckpt models_clauses/pg_best.pt --battles 100 --rollouts 3 --clauses` (`--opponent smart`, `--sweep`)
- **Search-judged team evolution:** `uv run python evolve_teams.py --ckpt models_clauses/pg_best.pt --pilots search --pop 12 --gens 5 --games 3 --rollouts 3 --clauses --out evolved_search` (slow — search every move)
- **Gotchas:** import order matters — `cinnabar.engine_cpp` must be imported before `import
  cinnabar_engine` (engine_cpp inserts `engine/build` on `sys.path`). Engine changes need
  `cmake --build build` + `ctest`. Search is **engine-path only** (needs `ce.Battle` to clone); the
  browser path would need state reconstruction first. The sandbox has g++ + python but no
  cmake/torch/poke-env — Conor builds/runs on his Mac and pastes output.

---

## One-paragraph narrative (the through-line)

We modeled Sleep/Freeze Clause and got a measurable agent gain (+27 Elo). We then attacked team
construction three ways — single-pilot coevolution, an anchored pilot panel, and co-training — and
all three produced off-meta junk (Snorlax/Exeggutor cut). That triangulated the real bottleneck: the
**judge** (the policy evaluating teams) was too weak to value good cores. Diagnostics then *refuted*
every specific weakness hypothesis (switch-loop, defense, stall) — the agent is a fine average player;
its holes only a strong human exposes. The one lever that survived measurement was **decision-time
search** (+19–27%). Distilling search into the policy (expert iteration) **failed**, but *using*
search directly as the team judge **worked** — the good cores came back. The remaining gap, with
Exeggutor as its clearest symptom, is **strategic/positional depth**, which 1-ply search and the
current policy don't have. That's the frontier the next session inherits.
