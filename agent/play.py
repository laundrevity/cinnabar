"""Phase 0 entrypoint: run the bot and let a human challenge it in the browser.

Usage (with a local Showdown server already running on :8000):

    cd agent
    python play.py                 # accept challenges from anyone, forever
    python play.py --opponent Foo  # only accept challenges from user "Foo"

Then open http://localhost:8000, pick a name, build/import a [Gen 1] OU team,
search for the user "CinnabarBot", and challenge it to a [Gen 1] OU battle.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from poke_env import AccountConfiguration

from cinnabar.policy import RandomPolicy
from cinnabar.showdown import PolicyPlayer

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_PATH = REPO_ROOT / "teams" / "gen1ou-sample.txt"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cinnabar bot vs a human.")
    parser.add_argument("--username", default="CinnabarBot")
    parser.add_argument(
        "--opponent",
        default=None,
        help="Only accept challenges from this username (default: anyone).",
    )
    parser.add_argument("--format", default="gen1ou", dest="battle_format")
    args = parser.parse_args()

    team = TEAM_PATH.read_text()
    bot = PolicyPlayer(
        policy=RandomPolicy(),
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.battle_format,
        team=team,
        start_timer_on_battle_start=True,  # don't stall waiting on a human
        max_concurrent_battles=1,
    )

    who = args.opponent or "anyone"
    print(
        f"{args.username} online. Challenge '{args.username}' to a "
        f"[{args.battle_format}] battle at http://localhost:8000  "
        f"(accepting from: {who}). Ctrl-C to stop."
    )
    while True:
        await bot.accept_challenges(args.opponent, 1)


if __name__ == "__main__":
    asyncio.run(main())
