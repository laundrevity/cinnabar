"""Ceiling probe: how much skill room exists ABOVE max-damage on a team set?

Pits SmartHeuristicPolicy (max-damage + sleep/paralysis/heal/pivot) against MaxDamagePolicy,
both non-mirror and MIRROR (same team both sides — removes matchup luck, isolates skill).
Pure policies + C++ engine; no torch.

    cd agent
    uv run python heuristic_probe.py                 # both pools, 300 battles
    uv run python heuristic_probe.py --battles 600

Read it as: if `smart vs maxdamage MIRROR` is ~50%, max-damage is near-optimal on these teams
as modeled -> the bottleneck is engine breadth (stubbed strategic moves), not the RL agent.
If it's clearly >55-60%, the room is real and the RL agent is the thing failing to find it.
The `smart vs random` / `maxdmg vs random` rows are a competence sanity check.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from cinnabar.engine_cpp import StaticData, load_teams, play_battle  # inserts engine/build on sys.path
import cinnabar_engine as ce  # noqa: E402
from cinnabar.policy import MaxDamagePolicy, RandomPolicy, SmartHeuristicPolicy  # noqa: E402
from train_engine import _FALLBACK_TEAMS  # noqa: E402  (2-team set; import pulls torch, harmless)


def _wlt(p1, p2, teams, n, base, mirror, static, turn_limit):
    w = lo = t = 0
    for i in range(n):
        t1 = random.choice(teams)
        t2 = t1 if mirror else random.choice(teams)
        b = play_battle(p1, p2, t1, t2, static, base + i, tag=f"{base}_{i}", turn_limit=turn_limit)
        r = b.result()
        if r == ce.Result.P1Win:
            w += 1
        elif r == ce.Result.P2Win:
            lo += 1
        else:
            t += 1
    f = 100.0 / max(n, 1)
    return w * f, lo * f, t * f


def _row(label, p1, p2, teams, n, base, mirror, static, turn_limit):
    w, lo, t = _wlt(p1, p2, teams, n, base, mirror, static, turn_limit)
    tag = "MIRROR    " if mirror else "non-mirror"
    print(f"    {label:22s} {tag} | W {w:5.1f}%  L {lo:5.1f}%  T {t:4.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Measure skill room above max-damage.")
    p.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    p.add_argument("--battles", type=int, default=300)
    p.add_argument("--turn-limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)

    static = StaticData(1)
    smart, md, rng = SmartHeuristicPolicy(), MaxDamagePolicy(), RandomPolicy()

    rulers = [("fallback 2-team", _FALLBACK_TEAMS)]
    loaded = load_teams(a.teams_dir)
    if loaded:
        rulers.append((f"{a.teams_dir.split('/')[-1]} ({len(loaded)} teams)", loaded))

    for label, teams in rulers:
        print(f"\n  pool: {label} | {a.battles} battles")
        base = 1
        _row("smart  vs maxdamage", smart, md, teams, a.battles, base, False, static, a.turn_limit)
        base += a.battles
        _row("smart  vs maxdamage", smart, md, teams, a.battles, base, True, static, a.turn_limit)
        base += a.battles
        _row("smart  vs random", smart, rng, teams, a.battles, base, False, static, a.turn_limit)
        base += a.battles
        _row("maxdmg vs random", md, rng, teams, a.battles, base, False, static, a.turn_limit)
        base += a.battles


if __name__ == "__main__":
    main()
