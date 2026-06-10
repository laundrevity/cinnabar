"""Tune the minimax search knobs OFFLINE — stop spending human games on knob settings.

Browser game 7 (paranoia=1.0, leaf-depth 0): a 60-turn switch carousel. Mechanism: pure worst-case
scoring never credits initiative — attacking carries the opponent's best reply as its score,
pivoting a fat resist carries only one incoming hit, so with switches legal every turn the
least-bad worst case is eternal rotation. The cure is the paranoia/leaf-depth blend, and THIS
script measures the cure without a human: for each config it plays engine battles and reports
win%, voluntary switch rate, and switch-streak tails (the carousel signature).

    cd agent
    uv run python probe_search_behavior.py --ckpt models_wf/pg_best.pt \
        --value-ckpt value_net/value_best.pt --battles 40 --clauses
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from cinnabar.engine_cpp import StaticData, load_teams
from cinnabar.policy import SmartHeuristicPolicy, StallerPolicy
from cinnabar.search import play_search_battle
from cinnabar.state import ActionType
from ladder import _load_net, _load_value

import cinnabar_engine as ce  # noqa: E402


class SwitchTracker:
    """Observer for play_search_battle: voluntary switch rate + longest streak + loopy battles."""

    def __init__(self) -> None:
        self.moves = 0
        self.switches = 0
        self.max_streak = 0
        self.loopy = 0
        self._streak = 0
        self._battle_max = 0

    def start_battle(self) -> None:
        if self._battle_max >= 5:
            self.loopy += 1
        self._streak = 0
        self._battle_max = 0

    def __call__(self, battle, s1, a1) -> None:
        if s1.force_switch:
            self._streak = 0
        elif a1.type == ActionType.SWITCH:
            self.switches += 1
            self._streak += 1
            self._battle_max = max(self._battle_max, self._streak)
            self.max_streak = max(self.max_streak, self._streak)
        else:
            self.moves += 1
            self._streak = 0

    def finish(self) -> None:
        self.start_battle()

    @property
    def rate(self) -> float:
        d = self.moves + self.switches
        return self.switches / d if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline minimax knob tuning (win% + switch behavior).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--value-ckpt", default=None)
    ap.add_argument("--battles", type=int, default=40, help="per config per opponent")
    ap.add_argument("--paranoia", default="0.0,0.5,1.0", help="comma list to sweep")
    ap.add_argument("--leaf-depth", default="0,8", help="comma list to sweep")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--opp-top-k", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=400)
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)
    teams = load_teams(a.teams_dir)

    net_policy = _load_net(a.ckpt, a.hidden, a.device, 1)
    net = net_policy.net
    if a.value_ckpt:
        from cinnabar.rl.net import HybridNet
        net = HybridNet(net, _load_value(a.value_ckpt, a.hidden, a.device))
    opp_model = SmartHeuristicPolicy()

    pick = random.Random(a.seed)
    matchups = [(pick.choice(teams), pick.choice(teams), 3000 + i) for i in range(a.battles)]
    paranoias = [float(x) for x in a.paranoia.split(",")]
    depths = [int(x) for x in a.leaf_depth.split(",")]
    opponents = [("smart", SmartHeuristicPolicy()), ("staller", StallerPolicy())]

    print(f"\nminimax knob sweep — {a.battles} paired battles/config/opponent, top-k {a.top_k}, "
          f"leaf {'ValueNet' if a.value_ckpt else 'PPO head'}, clauses {'on' if a.clauses else 'off'}\n")
    print(f"  {'config':22s} {'opponent':>8s} {'win%':>6s} {'switch%':>8s} {'maxstreak':>10s} {'loopy':>6s}")
    for p in paranoias:
        for d in depths:
            for opp_name, opp in opponents:
                tr = SwitchTracker()
                w = 0.0
                for t1, t2, s in matchups:
                    tr.start_battle()
                    r = play_search_battle(net, opp, opp_model, t1, t2, static, s, tag=f"pb{s}",
                                           turn_limit=a.turn_limit, clauses=a.clauses,
                                           device=a.device, rollouts=a.rollouts, top_k=a.top_k,
                                           minimax=True, opp_top_k=a.opp_top_k, paranoia=p,
                                           leaf_depth=d, rollout_policy=net_policy,
                                           observer=tr).result()
                    if r in (ce.Result.Tie, ce.Result.Ongoing):
                        w += 0.5
                    elif r == ce.Result.P1Win:
                        w += 1.0
                tr.finish()
                print(f"  p={p:.1f} leaf-depth={d:<3d} {opp_name:>8s} {w / a.battles * 100:5.1f} "
                      f"{tr.rate * 100:7.1f}% {tr.max_streak:10d} {tr.loopy:5d}/{a.battles}")


if __name__ == "__main__":
    main()
