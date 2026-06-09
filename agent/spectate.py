"""Watch two bot-piloted teams battle each other live in the browser.

Spins up two PolicyPlayers on the local Showdown server (e.g. two evolved teams, both piloted by
the trained net) and has them battle. Open http://localhost:8000 and click the battle under
"Watch a battle" (main menu) — `--delay` paces the moves so it's watchable in real time.

Browser-path caveat: only raw-net / heuristic / random pilots are available here — the search
pilot needs the live C++ engine battle to clone (state reconstruction from poke-env is the open
"browser ground truth" build). What you watch is the policy, not the search judge.

    cd agent
    # the top two evolved teams, net-piloted, paced for watching:
    uv run python spectate.py --ckpt models_cf/pg_best.pt \
        --p1 evolved_k3/evolved-01.txt --p2 evolved_k3/evolved-02.txt --delay 1.5
    # same team mirror, 3 battles, full speed (read the room log afterwards):
    uv run python spectate.py --ckpt models_cf/pg_best.pt \
        --p1 evolved_k3/evolved-01.txt --p2 evolved_k3/evolved-01.txt -n 3

Needs the local server running (../scripts/run-server.sh). gen1ou enforces Sleep/Freeze Clause,
matching --clauses training.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from poke_env import AccountConfiguration

from cinnabar.policy import RandomPolicy, SmartHeuristicPolicy, StallerPolicy
from cinnabar.showdown import PolicyPlayer


class PacedPlayer(PolicyPlayer):
    """PolicyPlayer with an optional per-move delay so humans can watch the battle live."""

    def __init__(self, *args, move_delay: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._move_delay = move_delay

    async def choose_move(self, battle):
        if self._move_delay:
            await asyncio.sleep(self._move_delay)
        return super().choose_move(battle)


def build_policy(kind: str, ckpt, hidden: int, device: str):
    if kind == "heuristic":
        return SmartHeuristicPolicy()
    if kind == "staller":
        return StallerPolicy()
    if kind == "random":
        return RandomPolicy()
    # raw trained net (greedy) — lazy imports so random/heuristic don't need torch
    if not ckpt:
        raise SystemExit("--pilot raw needs --ckpt")
    import torch

    from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
    from cinnabar.rl.agent import PGPolicy
    from cinnabar.rl.net import ActionScorer

    net = ActionScorer(GLOBAL_DIM, ACTION_DIM, hidden).to(device)
    sd = torch.load(ckpt, map_location=device)
    if (sd["policy_mlp.0.weight"].shape[1] != GLOBAL_DIM + ACTION_DIM
            or sd["value_mlp.0.weight"].shape[1] != GLOBAL_DIM):  # older checkpoint -> auto-pad
        from pad_checkpoint import pad_state_dict
        pad_state_dict(sd, 1)
    net.load_state_dict(sd)
    policy = PGPolicy(net, device=device)
    policy.eval()  # greedy
    return policy


def _username(prefix: str, team_path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]", "", Path(team_path).stem)
    return f"{prefix} {stem}"[:18]


async def main() -> None:
    ap = argparse.ArgumentParser(description="Bot-vs-bot battles on the local server, for spectating.")
    ap.add_argument("--p1", required=True, help="P1 team file (Showdown export .txt)")
    ap.add_argument("--p2", required=True, help="P2 team file")
    ap.add_argument("--ckpt", default=None, help="net checkpoint (for raw pilots)")
    ap.add_argument("--p1-pilot", choices=["raw", "heuristic", "staller", "random"], default="raw")
    ap.add_argument("--p2-pilot", choices=["raw", "heuristic", "staller", "random"], default="raw")
    ap.add_argument("-n", "--n-battles", type=int, default=1)
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds per move, so the battle is watchable live (0 = full speed; "
                         "needs a poke-env that accepts async choose_move — if it errors, use 0)")
    ap.add_argument("--format", default="gen1ou", dest="battle_format",
                    help="gen1ou enforces Sleep/Freeze Clause (matches --clauses training)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    pol1 = build_policy(a.p1_pilot, a.ckpt, a.hidden, a.device)
    pol2 = build_policy(a.p2_pilot, a.ckpt, a.hidden, a.device)
    cls = PacedPlayer if a.delay > 0 else PolicyPlayer
    extra = {"move_delay": a.delay} if a.delay > 0 else {}
    p1 = cls(policy=pol1, account_configuration=AccountConfiguration(_username("P1", a.p1), None),
             battle_format=a.battle_format, team=Path(a.p1).read_text(),
             max_concurrent_battles=1, **extra)
    p2 = cls(policy=pol2, account_configuration=AccountConfiguration(_username("P2", a.p2), None),
             battle_format=a.battle_format, team=Path(a.p2).read_text(),
             max_concurrent_battles=1, **extra)

    print(f"\n{p1.username} ({a.p1_pilot}, {Path(a.p1).name})  vs  "
          f"{p2.username} ({a.p2_pilot}, {Path(a.p2).name})")
    pace = f"{a.delay:.1f}s/move" if a.delay > 0 else "full speed"
    print(f"format {a.battle_format} | {a.n_battles} battle(s) | {pace}")
    print("\nOpen http://localhost:8000 -> 'Watch a battle' and click the room.\n")
    await p1.battle_against(p2, n_battles=a.n_battles)
    print(f"Done. {p1.username} won {p1.n_won_battles}/{p1.n_finished_battles}.")


if __name__ == "__main__":
    asyncio.run(main())
