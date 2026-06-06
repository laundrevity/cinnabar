"""Zero-friction smoke test: two random bots play each other in Gen 1 OU.

Run this first to confirm the whole stack works (Showdown + poke-env + team +
format) without anyone building a team by hand. Watch it live at
http://localhost:8000.

    cd agent
    python smoke_test.py            # 1 battle
    python smoke_test.py -n 20      # 20 battles
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cinnabar.policy import RandomPolicy
from cinnabar.showdown import PolicyPlayer

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_PATH = REPO_ROOT / "teams" / "gen1ou-sample.txt"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Bot-vs-bot smoke test.")
    parser.add_argument("-n", "--n-battles", type=int, default=1)
    args = parser.parse_args()

    team = TEAM_PATH.read_text()
    common = dict(battle_format="gen1ou", team=team, max_concurrent_battles=1)

    p1 = PolicyPlayer(policy=RandomPolicy(seed=1), **common)
    p2 = PolicyPlayer(policy=RandomPolicy(seed=2), **common)

    print(f"Running {args.n_battles} battle(s). Watch at http://localhost:8000 ...")
    await p1.battle_against(p2, n_battles=args.n_battles)
    print(f"Done. {p1.username} won {p1.n_won_battles}/{p1.n_finished_battles}.")


if __name__ == "__main__":
    asyncio.run(main())
