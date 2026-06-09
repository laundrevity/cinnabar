"""Measure decision-time search against the raw greedy policy — does lookahead help?

Paired comparison: for each battle the raw net and the search net play the SAME teams from the SAME
seed against the SAME opponent (as P1), so the difference is the decision rule, not luck. Reports
both win-rates against the staller (the hard case) and, optionally, the attacking heuristic.

    cd agent
    uv run python search_eval.py --ckpt models_clauses/pg_best.pt --battles 100 --rollouts 3 --clauses
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from cinnabar.engine_cpp import StaticData, load_teams, play_battle
from cinnabar.policy import SmartHeuristicPolicy, StallerPolicy
from cinnabar.search import play_search_battle
from ladder import _load_net

import cinnabar_engine as ce  # noqa: E402


def _p1_score(r) -> float:
    if r in (ce.Result.Tie, ce.Result.Ongoing):
        return 0.5
    return 1.0 if r == ce.Result.P1Win else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Decision-time search vs the raw policy.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--battles", type=int, default=100)
    ap.add_argument("--rollouts", type=int, default=3, help="dice samples per action in the lookahead")
    ap.add_argument("--opponent", choices=["staller", "smart"], default="staller")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)
    teams = load_teams(a.teams_dir)
    if not teams:
        raise SystemExit(f"no teams in {a.teams_dir}")

    net_policy = _load_net(a.ckpt, a.hidden, a.device, 1)  # greedy raw policy (and .net for search)
    net = net_policy.net
    opp = StallerPolicy() if a.opponent == "staller" else SmartHeuristicPolicy()
    opp_model = SmartHeuristicPolicy()  # the agent's ASSUMED opponent for the lookahead

    raw_w = srch_w = 0.0
    pick = random.Random(a.seed)
    for i in range(a.battles):
        t1, t2 = pick.choice(teams), pick.choice(teams)
        seed = 1000 + i
        r_raw = play_battle(net_policy, opp, t1, t2, static, seed, tag=f"r{seed}",
                            turn_limit=a.turn_limit, clauses=a.clauses).result()
        r_srch = play_search_battle(net, opp, opp_model, t1, t2, static, seed, tag=f"s{seed}",
                                    turn_limit=a.turn_limit, clauses=a.clauses, device=a.device,
                                    rollouts=a.rollouts).result()
        raw_w += _p1_score(r_raw)
        srch_w += _p1_score(r_srch)

    print(f"\n{Path(a.ckpt).name} vs {a.opponent} — {a.battles} paired battles, "
          f"rollouts {a.rollouts}, clauses {'on' if a.clauses else 'off'}\n")
    print(f"  raw policy win%     {raw_w / a.battles * 100:5.1f}")
    print(f"  search win%         {srch_w / a.battles * 100:5.1f}")
    print(f"  lift                {(srch_w - raw_w) / a.battles * 100:+5.1f}%  "
          f"(positive = lookahead helps)")


if __name__ == "__main__":
    main()
