"""Co-training: the agent and the teams improve against each other, with no injected priors.

Each round:
  1. EVOLVE teams against the current agent (the agent pilots both sides), scored against the
     growing team archive — the discovered meta so far, never a hand-picked set.
  2. ADD the round's teams to the archive.
  3. TRAIN the agent on the whole archive (warm-started from the current agent, league self-play),
     so it must learn to play every team discovered so far.
Repeat. Strategy — which teams are strong, how to pilot them, when two sleepers beat one — emerges
from the interaction instead of from hand-coded constraints or human team lists.

The only inputs are the game's rules (species/move legality) and win/loss. The starting agent is a
*policy* warm-start (not a team prior); the team archive is seeded with RANDOM teams, so the team
side starts from noise and bootstraps.

Collapse risk: alternating best-response can settle into a shared blind spot (agent and teams
co-adapt narrowly — e.g. neither ever learns to punish over-statusing because nothing in the loop
does it). Two no-prior diversity mechanisms guard against it: the archive GROWS (the agent must keep
beating the entire history, not just the latest teams) and league self-play keeps past agent
snapshots as opponents. The per-round ladder is the early-warning signal — if margin-over-smart
stalls while teams stay degenerate, the loop is collapsing, and the next step is PSRO (score
evolution against a population of past agents, not just the latest).

    cd agent
    uv run python cotrain.py --init models_clauses/pg_best.pt --rounds 6 --clauses --out cotrain
    uv run python cotrain.py --dry-run --rounds 2          # print the per-round commands only
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

from cinnabar import movesets

HERE = Path(__file__).resolve().parent


def to_showdown(team) -> str:
    return "\n\n".join(sp + "\n" + "\n".join("- " + m for m in mv) for sp, mv in team) + "\n"


def run(cmd, dry: bool) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, cwd=HERE, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Co-train the agent and teams against each other.")
    ap.add_argument("--init", default="models_clauses/pg_best.pt",
                    help="starting agent — a POLICY warm-start, not a team prior (pilots round 0)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed-teams", type=int, default=24,
                    help="random organic teams seeding round 0's archive (the team side starts from noise)")
    ap.add_argument("--evolve-pop", type=int, default=24)
    ap.add_argument("--evolve-gens", type=int, default=20)
    ap.add_argument("--evolve-keep", type=int, default=8, help="teams added to the archive per round")
    ap.add_argument("--train-iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--clauses", action="store_true", help="OU Sleep+Freeze Clause throughout")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ladder-battles", type=int, default=150, help="per-round ladder games/pair (0 to skip)")
    ap.add_argument("--out", default="cotrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the per-round commands without running")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out)
    archive = out / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    clause_flag = ["--clauses"] if a.clauses else []

    # Round 0 archive: random ORGANIC teams (no human teams). The team side bootstraps from noise.
    for i in range(a.seed_teams):
        team = [(sp, list(mv)) for sp, mv in movesets.generate_team(rng)]
        (archive / f"seed-{i:02d}.txt").write_text(to_showdown(team))
    print(f"seeded archive with {a.seed_teams} random teams -> {archive}")

    agent = a.init
    agents: list[str] = []
    for r in range(a.rounds):
        rd = out / f"round{r}"
        teams_out = rd / "teams"
        n_arch = len(list(archive.glob("*.txt")))

        # 1. evolve teams vs the current agent, scored against the growing archive
        print(f"\n=== round {r}: evolve teams (pilot={Path(agent).name}, archive={n_arch} teams) ===")
        run([sys.executable, "evolve_teams.py", "--ckpt", agent, "--pilots", "net",
             "--anchor-dir", str(archive), "--pop", str(a.evolve_pop), "--gens", str(a.evolve_gens),
             "--keep", str(a.evolve_keep), "--out", str(teams_out), "--hidden", str(a.hidden),
             "--device", a.device, "--seed", str(a.seed + r)] + clause_flag, a.dry_run)

        # 2. add the round's teams to the archive (grows every round)
        if not a.dry_run:
            for f in sorted(teams_out.glob("*.txt")):
                shutil.copy(f, archive / f"r{r}-{f.name}")

        # 3. train the agent on the grown archive, warm-started from the current agent
        agent_out = rd / "agent"
        print(f"\n=== round {r}: train agent on the archive ===")
        run([sys.executable, "train_engine.py", "--opponent", "league", "--reward", "shaped",
             "--teams-dir", str(archive), "--init", agent, "--iters", str(a.train_iters),
             "--batch", str(a.batch), "--anchor", "smart", "--anchor-frac", "0.5",
             "--snapshot-every", "10", "--out", str(agent_out), "--hidden", str(a.hidden),
             "--device", a.device, "--seed", str(a.seed + r)] + clause_flag, a.dry_run)
        agent = str(agent_out / "pg_best.pt")
        agents.append(agent)

        # 4. per-round ladder on the discovered meta — watch margin-over-smart across rounds
        if a.ladder_battles:
            print(f"\n=== round {r}: ladder (vs smart on the archive meta) ===")
            run([sys.executable, "ladder.py", "--teams-dir", str(archive), "--battles",
                 str(a.ladder_battles), "--hidden", str(a.hidden), "--device", a.device,
                 "--ckpts"] + agents + clause_flag, a.dry_run)

    print(f"\nco-training done.\n  final agent: {agent}\n  archive: {archive} "
          f"({len(list(archive.glob('*.txt')))} teams)")


if __name__ == "__main__":
    main()
