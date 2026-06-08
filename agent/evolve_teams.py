"""Team optimizer — evolve strong Gen 1 OU teams against a meta anchor, judged by a pilot panel.

The first cut (population-relative fitness, single net pilot) coadapted: it found teams that beat
*that net* by exploiting its quirks, not teams that are good. The heuristic re-rank caught it (the
net's #1 team was the heuristic's worst), and the teams were off-meta (no Snorlax, no sleeper). Two
root causes, both fixed here:

  1. **Fixed meta anchor.** Fitness is win-rate vs the hand-built `teams/` (real human teams), not vs
     the evolving population. "Good" now means "beats known-good teams," with an absolute reference —
     no more Red Queen churn where mean win-rate is pinned at 50% by construction.
  2. **Pilot panel.** Each team is scored under *several* pilots (the trained net AND the smart
     heuristic) and the results aggregated (mean, or `--agg min` for worst-case robustness). A team
     that only wins by farming one pilot's blind spot scores poorly under the other, so coadaptation
     is penalised instead of rewarded.

Plus the initial population is **seeded from the anchor teams**, so evolution starts from the human
meta core (big four + a sleep lead) and only drifts off it if that genuinely wins — it can't lose
the core by random initialisation. Genome / mutation / crossover (Species Clause, legal >=2-attack
sets, engine-modelled moves only) are unchanged.

    cd agent
    uv run python evolve_teams.py --ckpt models_clauses/pg_best.pt --pilots net,heuristic \
        --pop 24 --gens 30 --clauses --out evolved
    # heuristic-only (most meta-faithful judge while the net still switch-loops):
    uv run python evolve_teams.py --pilots heuristic --pop 24 --gens 30 --clauses --out evolved
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

# engine_cpp import puts engine/build on sys.path, so it must precede `import cinnabar_engine`.
from cinnabar import movesets
from cinnabar.engine_cpp import StaticData, load_teams, play_battle
from cinnabar.policy import SmartHeuristicPolicy
from ladder import _load_net

import cinnabar_engine as ce  # noqa: E402  (only importable after engine_cpp sets the path)

REPO_TEAMS = Path(__file__).resolve().parent.parent / "teams"
Team = list  # [(species, [moves]), ...]


# ----------------------------------------------------------------------------- genome operators
def random_team(rng: random.Random) -> Team:
    return [(sp, list(mv)) for sp, mv in movesets.generate_team(rng)]


def _legal_moveset(sp: str, mv: list[str]) -> bool:
    return sum(1 for m in mv if m not in movesets.STATUS_MOVES) >= 2  # keep >=2 attacks


def mutate(team: Team, rng: random.Random) -> Team:
    """One of: replace a species, re-roll a mon's moveset, or swap a single move."""
    t = [(sp, list(mv)) for sp, mv in team]
    i = rng.randrange(len(t))
    sp0 = t[i][0]
    roll = rng.random()
    if sp0 not in movesets.SPECIES_MOVEPOOLS:
        roll = 0.0  # a seeded team's off-pool species: only a full species-swap is safe
    if roll < 0.34:  # swap a whole species (keep Species Clause: distinct species)
        used = {sp for sp, _ in t}
        options = [s for s in movesets.ALL_SPECIES if s not in used]
        if options:
            sp = rng.choice(options)
            t[i] = (sp, movesets.sample_moveset(sp, rng))
    elif roll < 0.67:  # re-roll this mon's four moves
        t[i] = (sp0, movesets.sample_moveset(sp0, rng))
    else:  # swap a single move for another from the species' movepool
        mv = list(t[i][1])
        options = [m for m in movesets.SPECIES_MOVEPOOLS[sp0] if m not in mv]
        if options:
            mv[rng.randrange(len(mv))] = rng.choice(options)
            if _legal_moveset(sp0, mv):
                t[i] = (sp0, mv)
    return t


def crossover(a: Team, b: Team, rng: random.Random) -> Team:
    """Splice two parents: interleave their mons, keep the first of each species, until six."""
    pool = list(a) + list(b)
    rng.shuffle(pool)
    child: Team = []
    used: set[str] = set()
    for sp, mv in pool:
        if sp not in used:
            child.append((sp, list(mv)))
            used.add(sp)
        if len(child) == 6:
            break
    while len(child) < 6:  # top up if both parents shared species (rare)
        sp = rng.choice([s for s in movesets.ALL_SPECIES if s not in used])
        child.append((sp, movesets.sample_moveset(sp, rng)))
        used.add(sp)
    return child


def to_showdown(team: Team) -> str:
    return "\n\n".join(sp + "\n" + "\n".join("- " + m for m in mv) for sp, mv in team) + "\n"


def tournament_pick(pop, fit, rng: random.Random, k: int) -> Team:
    cand = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(cand, key=lambda c: fit[c])]


# ----------------------------------------------------------------------------- fitness
def _winrate(team, pol, anchor, static, games, seed, clauses, turn_limit):
    """Win-rate of `team` (piloted by `pol`) vs random anchor teams (same pilot), both lead
    positions alternated so a P1 edge can't bias it. Returns (rate, next_seed)."""
    w = 0.0
    for _ in range(games):
        opp = random.choice(anchor)
        lead = seed % 2 == 0  # does `team` lead (P1)?
        p1t, p2t = (team, opp) if lead else (opp, team)
        r = play_battle(pol, pol, p1t, p2t, static, seed, tag=f"ev{seed}",
                        turn_limit=turn_limit, clauses=clauses).result()
        if r in (ce.Result.Tie, ce.Result.Ongoing):
            w += 0.5
        elif (r == ce.Result.P1Win) == lead:  # the side `team` played on won
            w += 1.0
        seed += 1
    return w / max(games, 1), seed


def evaluate(pop, pilots, anchor, static, games, base_seed, clauses, turn_limit, agg):
    """Fitness per team = aggregate over the pilot panel of its win-rate vs the anchor teams."""
    fit = []
    seed = base_seed
    for team in pop:
        rates = []
        for _name, pol in pilots:
            wr, seed = _winrate(team, pol, anchor, static, games, seed, clauses, turn_limit)
            rates.append(wr)
        fit.append(min(rates) if agg == "min" else sum(rates) / len(rates))
    return fit, seed


def _summary(team: Team) -> str:
    return ", ".join(sp for sp, _ in team)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve Gen 1 OU teams vs a meta anchor, panel-judged.")
    ap.add_argument("--ckpt", default=None, help="pilot net checkpoint (required if 'net' in --pilots)")
    ap.add_argument("--pilots", default="net,heuristic", help="comma list of pilots: net, heuristic")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--anchor-dir", default=str(REPO_TEAMS), help="fixed reference teams (fitness opponents)")
    ap.add_argument("--no-seed-anchor", dest="seed_anchor", action="store_false",
                    help="start from random teams instead of seeding the population with the anchor teams")
    ap.add_argument("--agg", choices=["mean", "min"], default="mean", help="aggregate panel pilots")
    ap.add_argument("--pop", type=int, default=24, help="population size")
    ap.add_argument("--gens", type=int, default=30, help="generations")
    ap.add_argument("--games", type=int, default=10, help="anchor battles per team per pilot per gen")
    ap.add_argument("--elite", type=int, default=4, help="top teams carried over unchanged")
    ap.add_argument("--mutate-rate", type=float, default=0.85)
    ap.add_argument("--tournament", type=int, default=3, help="tournament size for parent selection")
    ap.add_argument("--keep", type=int, default=6, help="top teams to write out")
    ap.add_argument("--final-games", type=int, default=40, help="battles/team/pilot for the final ranking")
    ap.add_argument("--clauses", action="store_true", help="evaluate under OU Sleep+Freeze Clause")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--out", default="evolved", help="dir to write the top teams (.txt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(seed_anchor=True)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    static = StaticData(1)

    pilots = []
    for nm in [s.strip() for s in a.pilots.split(",") if s.strip()]:
        if nm == "net":
            if not a.ckpt:
                raise SystemExit("--pilots includes 'net' but no --ckpt was given")
            pilots.append(("net", _load_net(a.ckpt, a.hidden, a.device, 1)))
        elif nm == "heuristic":
            pilots.append(("heuristic", SmartHeuristicPolicy()))
        else:
            raise SystemExit(f"unknown pilot '{nm}' (use net / heuristic)")
    if not pilots:
        raise SystemExit("no pilots selected")

    anchor = load_teams(a.anchor_dir)
    if not anchor:
        raise SystemExit(f"no anchor teams found in {a.anchor_dir}")

    pop: list[Team] = []
    if a.seed_anchor:
        pop = [[(sp, list(mv)) for sp, mv in t] for t in anchor][:a.pop]
    while len(pop) < a.pop:
        pop.append(random_team(rng))

    print(f"evolving: pop={a.pop} gens={a.gens} games/team/pilot={a.games} agg={a.agg} "
          f"clauses={'on' if a.clauses else 'off'}\n"
          f"  pilots: {', '.join(n for n, _ in pilots)} | anchor: {len(anchor)} teams from {a.anchor_dir}\n"
          f"  seed-from-anchor: {'on' if a.seed_anchor else 'off'}\n")
    base = 1
    for g in range(a.gens):
        fit, base = evaluate(pop, pilots, anchor, static, a.games, base, a.clauses, a.turn_limit, a.agg)
        order = sorted(range(len(pop)), key=lambda k: -fit[k])
        pop = [pop[k] for k in order]
        fit = [fit[k] for k in order]
        mean = sum(fit) / len(fit)
        print(f"gen {g:3d}  best {fit[0]*100:5.1f}%  mean {mean*100:5.1f}%  | {_summary(pop[0])}")

        nxt = [pop[k] for k in range(min(a.elite, len(pop)))]  # elitism
        while len(nxt) < a.pop:
            child = crossover(tournament_pick(pop, fit, rng, a.tournament),
                              tournament_pick(pop, fit, rng, a.tournament), rng)
            if rng.random() < a.mutate_rate:
                child = mutate(child, rng)
            nxt.append(child)
        pop = nxt

    # Final ranking by the aggregate, then a per-pilot breakdown so any net/heuristic disagreement
    # (the coadaptation tell) is visible per finalist instead of hidden inside the aggregate.
    fit, base = evaluate(pop, pilots, anchor, static, a.final_games, base, a.clauses, a.turn_limit, a.agg)
    order = sorted(range(len(pop)), key=lambda k: -fit[k])
    pop = [pop[k] for k in order]
    fit = [fit[k] for k in order]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = "  ".join(f"{n:>9s}" for n, _ in pilots)
    print(f"\nFinal leaderboard — win% vs anchor, {a.final_games} games/pilot ({a.agg} agg):\n")
    print(f"  {'#':>2s} {'agg':>5s}  {cols}  team")
    for rank in range(min(a.keep, len(pop))):
        (out / f"evolved-{rank + 1:02d}.txt").write_text(to_showdown(pop[rank]))
        per = []
        for _name, pol in pilots:
            wr, base = _winrate(pop[rank], pol, anchor, static, a.final_games, base, a.clauses, a.turn_limit)
            per.append(wr)
        cells = "  ".join(f"{p*100:8.1f}%" for p in per)
        print(f"  {rank + 1:2d} {fit[rank]*100:4.0f}%  {cells}  {_summary(pop[rank])}")
    print(f"\nwrote top {min(a.keep, len(pop))} teams to {out}/  "
          f"(play one with: play.py --teams-dir {out})")


if __name__ == "__main__":
    main()
