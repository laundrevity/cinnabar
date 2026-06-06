"""Run the bot and let a human challenge it in the browser.

    cd agent
    python play.py                                          # random bot (Phase 0)
    python play.py --checkpoint models_sp/pg_best.pt        # your trained agent

Then open http://localhost:8000, build/import a [Gen 1] OU team, and challenge the
user 'CinnabarBot' to a [Gen 1] OU battle (it won't appear in any list — challenge
it by name). Needs a local Showdown server (../scripts/run-server.sh).
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


def build_policy(args):
    """RandomPolicy by default, or a trained PG net (greedy) from a checkpoint."""
    if not args.checkpoint:
        return RandomPolicy(), "random"
    # Lazy imports so the random bot doesn't require torch.
    import torch

    from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
    from cinnabar.rl.agent import PGPolicy
    from cinnabar.rl.net import ActionScorer

    net = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    policy = PGPolicy(net, device=args.device)
    policy.eval()  # greedy
    return policy, f"trained[{Path(args.checkpoint).name}]"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cinnabar bot vs a human.")
    parser.add_argument("--username", default="CinnabarBot")
    parser.add_argument("--opponent", default=None,
                        help="only accept challenges from this username (default: anyone)")
    parser.add_argument("--format", default="gen1ou", dest="battle_format")
    parser.add_argument("--checkpoint", default=None,
                        help="PG checkpoint to play (default: random bot)")
    parser.add_argument("--hidden", type=int, default=64, help="must match the trained net")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy, kind = build_policy(args)
    team = TEAM_PATH.read_text()
    bot = PolicyPlayer(
        policy=policy,
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.battle_format,
        team=team,
        start_timer_on_battle_start=True,  # don't stall waiting on a human
        max_concurrent_battles=1,
    )

    who = args.opponent or "anyone"
    print(
        f"\n{args.username} ({kind}) is online (format: {args.battle_format}, accepting from: {who}).\n"
        f"\nIt won't appear in any user list — challenge it by name:\n"
        f"  1. Open http://localhost:8000 and build a {args.battle_format} team in the\n"
        f"     Teambuilder (you can paste teams/gen1ou-sample.txt).\n"
        f"  2. Type this into any chat box (e.g. the Lobby):\n"
        f"         /challenge {args.username}, {args.battle_format}\n"
        f"\nThe bot auto-accepts. Ctrl-C to stop.\n"
    )
    while True:
        await bot.accept_challenges(args.opponent, 1)


if __name__ == "__main__":
    asyncio.run(main())
