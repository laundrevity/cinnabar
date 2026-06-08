"""Coevolutionary team optimizer — evolve strong Gen 1 OU teams.

Uniform team generation (`--gen-teams`) spends nearly all its effort on junk teams: the team space
is astronomical and almost all of it is bad. This *evolves* a population of teams toward ones that
WIN. Fitness is measured by our trained agent piloting BOTH sides — so the only variable is the team
— against the rest of the population. As the population improves, fitness becomes relative to the
current strong teams: a self-forming metagame. The first, tractable cut at team construction (the
agent building its own team), before a learned drafter.

    cd agent
    # needs a checkpoint at the CURRENT GLOBAL_DIM (models_clauses, or pad an older one):
    #   uv run python pad_checkpoint.py models_genteams2/pg_best.pt /tmp/gt2_168.pt 128 1
    uv run python evolve_teams.py --ckpt models_clauses/pg_best.pt --pop 24 --gens 30 \
        --clauses --out evolved

Genome = 6 distinct species (Species Clause), each with a moveset from cinnabar/movesets.py (only
engine-modelled moves, so every evolved team simulates bit-for-bit). Mutation swaps a species /
re-rolls a moveset / swaps a single move; crossover splices two parents. Output: the top teams as
Showdown .txt (drop them into teams/) plus a win-rate leaderboard.

Caveat — coadaptation: evolution optimises teams for THIS pilot, so it can exploit the net's quirks
rather than find universally strong teams. Use the strongest, most general (gen-teams-trained) net,
and sanity-check the winners with `--smoke-vs-smart` (re-rank the finalists with the heuristic pilot).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

import cinnabar_engine as ce
from cinnabar import movesets
from cinnabar.engine_cpp import StaticData, play_battle
from cinnabar.policy import SmartHeuristicPolicy
from ladder import _load_net

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
    roll = rng.random()
    if roll < 0.34:  # swap a whole species (keep Species Clause: distinct species)
        used = {sp for sp, _ in t}
        options = [s for s in movesets.ALL_SPECIES if s not in used]
        if options:
            sp = rng.choice(options)
            t[i] = (sp, movesets.sample_moveset(sp, rng))
    elif roll < 0.67:  # re-roll this mon's four moves
        sp = t[i][0]
        t[i] = (sp, movesets.sample_moveset(sp, rng))
    else:  # swap a single move for another from the species' movepool
        sp, mv = t[i][0], list(t[i][1])
        options = [m for m in movesets.SPECIES_MOVEPOOLS[sp] if m not in mv]
        if options:
            mv[rng.randrange(len(mv))] = rng.choice(options)
            if _legal_moveset(sp, mv):
                t[i] = (sp, mv)
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


# ----------------------------------------------------------------------------- fitness
def evaluate(pop, p1, p2, static, games_per_team, base_seed, clauses, turn_limit):
    """Round-robin-ish: ~games_per_team battles per team vs random opponents, both sides piloted by
    the same policy so only the team differs. Returns (fitness win-rates, next base_seed)."""
    n = len(pop)
    wins = [0.0] * n
    games = [0.0] * n
    n_battles = max(1, n * games_per_team // 2)
    seed = base_seed
    for _ in range(n_battles):
        i, j = random.sample(range(n), 2)
        lead = i if seed % 2 == 0 else j  # alternate who leads (P1) to cancel any first-move edge
        other = j if lead == i else i
        r = play_battle(p1, p2, pop[lead], pop[other], static, seed,
                        tag=f"ev{seed}", turn_limit=turn_limit, clauses=clauses).result()
        games[i] += 1.0
        games[j] += 1.0
        if r in (ce.Result.Tie, ce.Result.Ongoing):
            wins[i] += 0.5
            wins[j] += 0.5
        elif r == ce.Result.P1Win:
            wins[lead] += 1.0
        else:
            wins[other] += 1.0
        seed += 1
    fitness = [wins[k] / max(games[k], 1.0) for k in range(n)]
    return fitness, seed


def _summary(team: Team) -> str:
    return ", ".join(sp for sp, _ in team)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve strong Gen 1 OU teams (coevolution).")
    ap.add_argument("--ckpt", required=True, help="pilot net checkpoint (current GLOBAL_DIM)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pop", type=int, default=24, help="population size")
    ap.add_argument("--gens", type=int, default=30, help="generations")
    ap.add_argument("--games", type=int, default=10, help="fitness battles per team per generation")
    ap.add_argument("--elite", type=int, default=4, help="top teams carried over unchanged")
    ap.add_argument("--mutate-rate", type=float, default=0.85)
    ap.add_argument("--tournament", type=int, default=3, help="tournament size for parent selection")
    ap.add_argument("--keep", type=int, default=6, help="top teams to write out")
    ap.add_argument("--final-games", type=int, default=40, help="battles/team for the final ranking")
    ap.add_argument("--clauses", action="store_true", help="evaluate under OU Sleep+Freeze Clause")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--out", default="evolved", help="dir to write the top teams (.txt)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    static = StaticData(1)
    net = _load_net(a.ckpt, a.hidden, a.device, 1)  # greedy pilot, both sides

    pop = [random_team(rng) for _ in range(a.pop)]
    base = 1
    print(f"evolving: pop={a.pop} gens={a.gens} games/team={a.games} "
          f"clauses={'on' if a.clauses else 'off'} pilot={Path(a.ckpt).name}\n")
    for g in range(a.gens):
        fit, base = evaluate(pop, net, net, static, a.games, base, a.clauses, a.turn_limit)
        order = sorted(range(len(pop)), key=lambda k: -fit[k])
        pop = [pop[k] for k in order]
        fit = [fit[k] for k in order]
        mean = sum(fit) / len(fit)
        print(f"gen {g:3d}  best {fit[0]*100:5.1f}%  mean {mean*100:5.1f}%  | {_summary(pop[0])}")

        nxt = [pop[k] for k in range(min(a.elite, len(pop)))]  # elitism
        while len(nxt) < a.pop:
            def pick():
                cand = random.sample(range(len(pop)), min(a.tournament, len(pop)))
                return pop[max(cand, key=lambda c: fit[c])]
            child = crossover(pick(), pick(), rng)
            if rng.random() < a.mutate_rate:
                child = mutate(child, rng)
            nxt.append(child)
        pop = nxt

    # Final, higher-precision ranking of the surviving population.
    fit, base = evaluate(pop, net, net, static, a.final_games, base, a.clauses, a.turn_limit)
    order = sorted(range(len(pop)), key=lambda k: -fit[k])
    pop = [pop[k] for k in order]
    fit = [fit[k] for k in order]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nFinal leaderboard (win% vs final population, {a.final_games} games/team):\n")
    for rank in range(min(a.keep, len(pop))):
        path = out / f"evolved-{rank + 1:02d}.txt"
        path.write_text(to_showdown(pop[rank]))
        print(f"  {rank + 1:2d}. {fit[rank]*100:5.1f}%  {_summary(pop[rank])}")
    print(f"\nwrote top {min(a.keep, len(pop))} teams to {out}/  (drop into teams/ to train on them)")

    # Optional sanity pass: re-rank the finalists with the heuristic pilot (coadaptation check).
    if a.keep and len(pop) >= 2:
        smart = SmartHeuristicPolicy()
        sfit, _ = evaluate(pop[:a.keep], smart, smart, static, a.final_games, base, a.clauses, a.turn_limit)
        print("\nSame finalists, piloted by the heuristic (sanity vs coadaptation):")
        for rank in range(min(a.keep, len(pop))):
            print(f"  {rank + 1:2d}. {sfit[rank]*100:5.1f}%  {_summary(pop[rank])}")


if __name__ == "__main__":
    main()
