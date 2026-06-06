"""Phase 1 yardstick: win rate of one policy against another over N games.

Needs a local Showdown server running (../scripts/run-server.sh).

    cd agent
    uv run python evaluate.py                       # maxdamage vs random, 100 games
    uv run python evaluate.py -n 300 --a maxdamage --b random
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cinnabar.policy import MaxDamagePolicy, RandomPolicy
from cinnabar.showdown import PolicyPlayer

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM = (REPO_ROOT / "teams" / "gen1ou-sample.txt").read_text()

POLICIES = {"random": RandomPolicy, "maxdamage": MaxDamagePolicy}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one policy vs another.")
    parser.add_argument("-n", "--n-battles", type=int, default=100)
    parser.add_argument("--a", default="maxdamage", choices=POLICIES)
    parser.add_argument("--b", default="random", choices=POLICIES)
    args = parser.parse_args()

    common = dict(battle_format="gen1ou", team=TEAM, max_concurrent_battles=10)
    a = PolicyPlayer(policy=POLICIES[args.a](), **common)
    b = PolicyPlayer(policy=POLICIES[args.b](), **common)

    print(f"Running {args.n_battles} games: {args.a} vs {args.b} ...")
    await a.battle_against(b, n_battles=args.n_battles)

    n = max(a.n_finished_battles, 1)
    pct = 100 * a.n_won_battles / n
    print(f"{args.a} won {a.n_won_battles}/{a.n_finished_battles} ({pct:.1f}%) vs {args.b}")


if __name__ == "__main__":
    asyncio.run(main())
