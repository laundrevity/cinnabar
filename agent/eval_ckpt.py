"""Controlled evaluation of a checkpoint — pin the team set AND the decoding so numbers
are comparable across runs and checkpoints.

Why this exists: ``train_engine.eval_winrate`` evaluates against the *global* ``TEAMS``
pool, so changing the team set (e.g. 2 fallback teams -> 5 parsed teams) silently changes
the benchmark. A "63% -> 50%" move can then be a change of ruler, not of skill. This script
fixes the ruler: a chosen team pool, greedy *and* sampled decoding, mirror option, and an
explicit Win/Loss/Tie breakdown (ties are turn-limit stalls and get hidden by a raw win%).

    cd agent
    # current best on BOTH pools — does it hold ~63% on the original 2-team ruler?
    uv run python eval_ckpt.py --init models_engine/pg_best.pt
    # isolate policy skill from team-matchup luck (same team both sides):
    uv run python eval_ckpt.py --init models_engine/pg_best.pt --mirror
    # compare two checkpoints on the same ruler:
    uv run python eval_ckpt.py --init A.pt B.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import torch

import train_engine as T  # reuse _STATIC, _run, build_state, select_batch machinery
import cinnabar_engine as ce
from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
from cinnabar.engine_cpp import StaticData, load_teams
from cinnabar.policy import MaxDamagePolicy, RandomPolicy, SmartHeuristicPolicy
from cinnabar.rl.net import ActionScorer


def _make_items(teams, n, base, mirror):
    items = []
    for i in range(n):
        t1 = random.choice(teams)
        t2 = t1 if mirror else random.choice(teams)
        s1 = [(s, list(m)) for s, m in t1]
        s2 = [(s, list(m)) for s, m in t2]
        items.append({"b": ce.make_battle(s1, s2, base + i), "s1": s1, "s2": s2, "tag": f"{base}_{i}"})
    return items


def _wlt(net, opp, args, teams, n, base, *, greedy, mirror):
    """Return (win%, loss%, tie%) for net (P1) vs opp (P2) over n battles."""
    items = _make_items(teams, n, base, mirror)
    T._run(items, net, opp, args, record_buf=None, greedy_learner=greedy)
    w = sum(1 for it in items if it["b"].result() == ce.Result.P1Win)
    loss = sum(1 for it in items if it["b"].result() == ce.Result.P2Win)
    t = n - w - loss
    f = 100.0 / max(n, 1)
    return w * f, loss * f, t * f


def _report(net, teams, label, args, base, *, mirror):
    md, rng, sm = MaxDamagePolicy(), RandomPolicy(), SmartHeuristicPolicy()
    print(f"  pool: {label} | {args.battles} battles{' | MIRROR (same team both sides)' if mirror else ''}")
    for name, opp, modes in (("smart", sm, ("greedy", "sampled")),
                             ("maxdamage", md, ("greedy", "sampled")),
                             ("random", rng, ("greedy",))):
        for mode in modes:
            w, l, t = _wlt(net, opp, args, teams, args.battles, base, greedy=(mode == "greedy"), mirror=mirror)
            base += 1
            print(f"    vs {name:9s} {mode:8s} | W {w:5.1f}%  L {l:5.1f}%  T {t:4.1f}%")
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="Controlled checkpoint eval (pinned teams + decoding).")
    p.add_argument("init", nargs="*", help="checkpoint(s) to evaluate")
    p.add_argument("--init", dest="init_flag", nargs="*", default=None, help="checkpoint(s) (alias)")
    p.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    p.add_argument("--battles", type=int, default=300)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--turn-limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mirror", action="store_true", help="same team on both sides (isolates skill)")
    p.add_argument("--fallback-only", action="store_true", help="only the 2-team fallback ruler")
    p.add_argument("--teams-only", action="store_true", help="only the teams/ ruler")
    a = p.parse_args()

    ckpts = list(a.init) + list(a.init_flag or [])
    if not ckpts:
        p.error("give at least one checkpoint, e.g. eval_ckpt.py models_engine/pg_best.pt")

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    T._STATIC = StaticData(1)
    args = SimpleNamespace(device=a.device, turn_limit=a.turn_limit, battles=a.battles)

    rulers = []
    if not a.teams_only:
        rulers.append(("fallback 2-team", T._FALLBACK_TEAMS))
    if not a.fallback_only:
        loaded = load_teams(a.teams_dir)
        if loaded:
            rulers.append((f"{a.teams_dir} ({len(loaded)} teams)", loaded))

    for ck in ckpts:
        net = ActionScorer(GLOBAL_DIM, ACTION_DIM, a.hidden).to(a.device)
        net.load_state_dict(torch.load(ck, map_location=a.device))
        print(f"\n=== {ck} ===")
        base = 1
        for label, teams in rulers:
            base = _report(net, teams, label, args, base, mirror=a.mirror) + 1000


if __name__ == "__main__":
    main()
