"""Win rate of one policy against another over N games.

Compare baselines, or evaluate a trained RL checkpoint:

    cd agent
    uv run python evaluate.py                                              # maxdamage vs random
    uv run python evaluate.py --a pg --b maxdamage --checkpoint models/pg_best.pt -n 500

Needs a local Showdown server (../scripts/run-server.sh). Use a large -n (e.g. 500)
for a tight estimate — a 50-game eval has a ~±14% confidence interval.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cinnabar.policy import MaxDamagePolicy, RandomPolicy
from cinnabar.showdown import PolicyPlayer
from cinnabar.teams import build_random_teambuilder, load_team_strings

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = REPO_ROOT / "teams"
CHOICES = ["random", "maxdamage", "pg"]


def build_policy(name: str, args: argparse.Namespace):
    if name == "random":
        return RandomPolicy()
    if name == "maxdamage":
        return MaxDamagePolicy()
    if name == "pg":
        # Lazy imports so baseline-only evals don't require torch.
        import torch

        from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
        from cinnabar.rl.agent import PGPolicy
        from cinnabar.rl.net import ActionScorer

        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when a side is 'pg'")
        net = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        policy = PGPolicy(net, device=args.device)
        policy.eval()  # greedy, deterministic, no trajectory recording
        return policy
    raise ValueError(name)


async def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate one policy vs another.")
    p.add_argument("-n", "--n-battles", type=int, default=100)
    p.add_argument("--a", default="maxdamage", choices=CHOICES)
    p.add_argument("--b", default="random", choices=CHOICES)
    p.add_argument("--checkpoint", default=None, help="PG checkpoint path (when a side is 'pg')")
    p.add_argument("--hidden", type=int, default=64, help="must match the trained net")
    p.add_argument("--device", default="cpu")
    p.add_argument("--teams-dir", default=str(TEAMS_DIR), help="dir of team .txt files (random per battle)")
    args = p.parse_args()

    teambuilder = build_random_teambuilder(load_team_strings(args.teams_dir))
    common = dict(battle_format="gen1ou", team=teambuilder, max_concurrent_battles=10)
    a = PolicyPlayer(policy=build_policy(args.a, args), **common)
    b = PolicyPlayer(policy=build_policy(args.b, args), **common)

    label_a = args.a if args.a != "pg" else f"pg[{Path(args.checkpoint).name}]"
    print(f"Running {args.n_battles} games: {label_a} vs {args.b} ...")
    await a.battle_against(b, n_battles=args.n_battles)

    finished = max(a.n_finished_battles, 1)
    losses = getattr(a, "n_lost_battles", finished - a.n_won_battles)
    ties = a.n_finished_battles - a.n_won_battles - losses
    pct = 100 * a.n_won_battles / finished
    print(f"{label_a}: {a.n_won_battles}/{a.n_finished_battles} = {pct:.1f}% vs {args.b}  [ties: {ties}]")


if __name__ == "__main__":
    asyncio.run(main())
