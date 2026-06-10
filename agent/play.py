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
from cinnabar.teams import build_random_teambuilder, load_team_strings

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = REPO_ROOT / "teams"


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
    sd = torch.load(args.checkpoint, map_location=args.device)
    if (sd["policy_mlp.0.weight"].shape[1] != GLOBAL_DIM + ACTION_DIM
            or sd["value_mlp.0.weight"].shape[1] != GLOBAL_DIM):  # older checkpoint -> auto-pad
        from pad_checkpoint import pad_state_dict
        pad_state_dict(sd, 1)
    net.load_state_dict(sd)
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
    parser.add_argument("--hidden", type=int, default=128, help="must match the trained net (current nets: 128)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teams-dir", default=str(TEAMS_DIR), help="dir of team .txt files (random per battle)")
    parser.add_argument("--search", action="store_true",
                        help="decide by k-gated decision-time search on a per-turn engine "
                             "reconstruction of the live battle (browser ground truth; needs "
                             "--checkpoint + the built engine). Falls back to the raw policy on "
                             "any reconstruction failure.")
    parser.add_argument("--top-k", type=int, default=3, help="policy-prior gating for --search")
    parser.add_argument("--rollouts", type=int, default=3, help="search rollouts per action")
    args = parser.parse_args()

    policy, kind = build_policy(args)
    team = build_random_teambuilder(load_team_strings(args.teams_dir))
    common = dict(
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=args.battle_format,
        team=team,
        start_timer_on_battle_start=True,  # don't stall waiting on a human
        max_concurrent_battles=1,
    )
    if args.search:
        if not args.checkpoint:
            raise SystemExit("--search needs --checkpoint (the net is the search prior + leaf)")
        from cinnabar.policy import SmartHeuristicPolicy
        from cinnabar.recon import SearchPolicyPlayer

        bot = SearchPolicyPlayer(policy=policy, net=policy.net, opp_model=SmartHeuristicPolicy(),
                                 rollouts=args.rollouts, top_k=args.top_k, clauses=True,
                                 device=args.device, **common)
        kind += f"+search[k={args.top_k},r={args.rollouts}]"
    else:
        bot = PolicyPlayer(policy=policy, **common)

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
