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
from ladder import _load_net, _load_value

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
    ap.add_argument("--leaf", choices=["value", "rollout"], default="value",
                    help="value: score the leaf with the value head (fast); rollout: play to terminal (deep, slow)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="policy-prior gating: search only the policy's top-k actions (0 = search all)")
    ap.add_argument("--value-ckpt", default=None,
                    help="calibrated win-prob ValueNet (train_value.py) as the search leaf; the "
                         "policy net still proposes (HybridNet). Default: the PPO value head.")
    ap.add_argument("--minimax", action="store_true",
                    help="adversarial lookahead: worst case over the opponent's replies "
                         "(vs the default point-estimate opponent model)")
    ap.add_argument("--paranoia", type=float, default=1.0)
    ap.add_argument("--opp-top-k", type=int, default=0)
    ap.add_argument("--opp-temp", type=float, default=0.0,
                    help="quantal-response opponent weighting (overrides --paranoia)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep rollouts {1,3,6} value-leaf, a rollout-leaf config, and top-k {3,4} "
                         "to map the headroom (slow — use a small --battles, e.g. 30)")
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
    if a.value_ckpt:
        from cinnabar.rl.net import HybridNet
        net = HybridNet(net, _load_value(a.value_ckpt, a.hidden, a.device))
        print(f"  search leaf: calibrated ValueNet {Path(a.value_ckpt).name} (policy net proposes)")
    opp = StallerPolicy() if a.opponent == "staller" else SmartHeuristicPolicy()
    opp_model = SmartHeuristicPolicy()  # the agent's ASSUMED opponent for the lookahead

    # Fix the teams + seeds once; raw and every search config play the identical matchups (paired).
    pick = random.Random(a.seed)
    matchups = [(pick.choice(teams), pick.choice(teams), 1000 + i) for i in range(a.battles)]
    raw_w = sum(_p1_score(play_battle(net_policy, opp, t1, t2, static, s, tag=f"r{s}",
                                      turn_limit=a.turn_limit, clauses=a.clauses).result())
                for t1, t2, s in matchups)

    def eval_search(rollouts, leaf, top_k):
        w = 0.0
        stats: dict = {}
        rp = net_policy if leaf == "rollout" else None
        for t1, t2, s in matchups:
            r = play_search_battle(net, opp, opp_model, t1, t2, static, s, tag=f"s{s}",
                                   turn_limit=a.turn_limit, clauses=a.clauses, device=a.device,
                                   rollouts=rollouts, leaf=leaf, rollout_policy=rp, stats=stats,
                                   top_k=top_k, minimax=a.minimax, opp_top_k=a.opp_top_k,
                                   paranoia=a.paranoia, opp_temp=a.opp_temp).result()
            w += _p1_score(r)
        return w, stats

    n = a.battles
    print(f"\n{Path(a.ckpt).name} vs {a.opponent} — {n} paired battles, clauses {'on' if a.clauses else 'off'}\n")
    print(f"  raw policy win%     {raw_w / n * 100:5.1f}\n")
    configs = ([(1, "value", 0), (3, "value", 0), (6, "value", 0), (3, "rollout", 0),
                (3, "value", 3), (3, "value", 4)]
               if a.sweep else [(a.rollouts, a.leaf, a.top_k)])
    print(f"  {'config':22s} {'search%':>8s} {'lift':>7s}  re-slept")
    for rollouts, leaf, top_k in configs:
        w, stats = eval_search(rollouts, leaf, top_k)
        rs = f"{stats.get('reslept', 0)}/{stats.get('tempt', 0)}"
        gate = f" k={top_k}" if top_k else " k=all"
        print(f"  r={rollouts} leaf={leaf:7s}{gate} {w / n * 100:7.1f}% {(w - raw_w) / n * 100:+6.1f}%   {rs}")


if __name__ == "__main__":
    main()
