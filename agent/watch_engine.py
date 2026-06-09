"""Watch engine battles as turn-by-turn transcripts — see what the judge actually does.

The evolution fitness, the search judge, the probes: all of it runs as silent C++ battles. This
renders them human-readable so a strong player can audit the play directly (e.g. watch an
evolved_k3 team piloted by gated search against an anchor team, and see whether the Explosions
and sleep targets make sense). The engine has no replay log; the transcript is reconstructed
in Python from `team_state` diffs around each step — same source of truth as the agent's view.

    cd agent
    # watch the #1 evolved team play a random anchor under the gated search judge:
    uv run python watch_engine.py --ckpt models_cf/pg_best.pt --p1 evolved_k3/evolved-01.txt \
        --clauses --top-k 3
    # specific matchup, raw-policy pilot vs the staller, 3 battles:
    uv run python watch_engine.py --ckpt models_cf/pg_best.pt --p1 evolved_k3/evolved-01.txt \
        --p2 ../teams/some_anchor.txt --p1-pilot raw --p2-pilot staller --battles 3 --clauses
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from cinnabar.engine_cpp import Reveal, StaticData, build_state, load_teams, parse_team, reveal_move
from cinnabar.policy import SmartHeuristicPolicy, StallerPolicy
from cinnabar.search import search_action_index
from ladder import _load_net

import cinnabar_engine as ce  # noqa: E402

STATUS_NAMES = {"SLP": "slp", "PAR": "par", "FRZ": "frz", "BRN": "brn", "PSN": "psn"}


def _mon(e) -> str:
    """'Snorlax 78% par' from a team_state entry (species, hp, status, fainted, active)."""
    s = f"{e[0]} {e[1] * 100:.0f}%"
    if e[2]:
        s += f" {STATUS_NAMES.get(e[2], e[2].lower())}"
    return s


def _deltas(side: str, prev, cur) -> list[str]:
    """Readable consequences of one step for one side: damage/heal, status, faints, switch-ins."""
    out = []
    for p, c in zip(prev, cur):
        if c[3] and not p[3]:
            out.append(f"{side} {p[0]} fainted")
            continue
        bits = []
        if abs(c[1] - p[1]) >= 0.005:
            bits.append(f"{p[1] * 100:.0f}%→{c[1] * 100:.0f}%")
        if c[2] != p[2]:
            bits.append(STATUS_NAMES.get(c[2], c[2].lower()) if c[2] else "status cured")
        if bits:
            out.append(f"{side} {p[0]} " + " ".join(bits))
        if c[4] and not p[4] and not c[3]:
            out.append(f"{side} sent out {c[0]}")
    return out


def make_picker(kind: str, net_policy, static, top_k: int, rollouts: int, device: str):
    """Returns pick(battle, player, state, my_spec, opp_spec, reveal) -> Action."""
    if kind == "search":
        net = net_policy.net

        def pick(battle, player, state, my_spec, opp_spec, reveal):
            i = search_action_index(battle, player, net, SmartHeuristicPolicy(), static,
                                    my_spec, opp_spec, reveal=reveal, device=device,
                                    rollouts=rollouts, state=state, top_k=top_k)
            return state.available_actions[i]
        return pick
    pol = {"raw": net_policy, "heuristic": SmartHeuristicPolicy(), "staller": StallerPolicy()}[kind]
    return lambda battle, player, state, my_spec, opp_spec, reveal: pol.select_action(state)


def watch(team1, team2, name1, name2, pick1, pick2, static, seed, clauses, turn_limit) -> None:
    spec1 = [(s, list(m)) for s, m in team1]
    spec2 = [(s, list(m)) for s, m in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    print(f"\n=== {name1}  vs  {name2}  (seed {seed}) ===")
    print(f"  P1: {', '.join(s for s, _ in spec1)}")
    print(f"  P2: {', '.join(s for s, _ in spec2)}\n")
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, "w", reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, "w_o", reveal=r2, opp_team=spec1)
        a1 = pick1(battle, 0, s1, spec1, spec2, r1)
        a2 = pick2(battle, 1, s2, spec2, spec1, r2)
        pre1, pre2 = battle.team_state(0), battle.team_state(1)
        act1 = next((e for e in pre1 if e[4]), None)
        act2 = next((e for e in pre2 if e[4]), None)
        print(f"T{battle.turn:3d}  {_mon(act1) if act1 else '---':24s} vs  {_mon(act2) if act2 else '---'}")
        print(f"      P1: {a1.label:20s} | P2: {a2.label}")
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
        for line in _deltas("P1", pre1, battle.team_state(0)) + _deltas("P2", pre2, battle.team_state(1)):
            print(f"      · {line}")
    res = battle.result()
    tag = {ce.Result.P1Win: f"{name1} (P1) wins", ce.Result.P2Win: f"{name2} (P2) wins",
           ce.Result.Tie: "tie"}.get(res, f"no result in {turn_limit} turns")
    print(f"\n  >>> {tag} — turn {battle.turn}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render engine battles as turn-by-turn transcripts.")
    ap.add_argument("--ckpt", default=None, help="net checkpoint (needed for raw/search pilots)")
    ap.add_argument("--p1", required=True, help="P1 team file (Showdown export .txt)")
    ap.add_argument("--p2", default=None, help="P2 team file; default = random anchor from --anchor-dir")
    ap.add_argument("--anchor-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--p1-pilot", choices=["search", "raw", "heuristic", "staller"], default="search")
    ap.add_argument("--p2-pilot", choices=["search", "raw", "heuristic", "staller"], default="search")
    ap.add_argument("--top-k", type=int, default=3, help="policy-prior gating for search pilots (0 = all)")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--battles", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)

    net_policy = None
    if "raw" in (a.p1_pilot, a.p2_pilot) or "search" in (a.p1_pilot, a.p2_pilot):
        if not a.ckpt:
            raise SystemExit("raw/search pilots need --ckpt")
        net_policy = _load_net(a.ckpt, a.hidden, a.device, 1)

    team1 = parse_team(Path(a.p1).read_text())
    name1 = Path(a.p1).stem
    if a.p2:
        p2_pool = [(parse_team(Path(a.p2).read_text()), Path(a.p2).stem)]
    else:
        anchors = load_teams(a.anchor_dir)
        if not anchors:
            raise SystemExit(f"no teams in {a.anchor_dir}")
        p2_pool = [(t, f"anchor{i:02d}") for i, t in enumerate(anchors)]

    pick1 = make_picker(a.p1_pilot, net_policy, static, a.top_k, a.rollouts, a.device)
    pick2 = make_picker(a.p2_pilot, net_policy, static, a.top_k, a.rollouts, a.device)
    print(f"P1 pilot: {a.p1_pilot}  |  P2 pilot: {a.p2_pilot}  |  top-k {a.top_k}  |  "
          f"clauses {'on' if a.clauses else 'off'}")
    for i in range(a.battles):
        team2, name2 = random.choice(p2_pool)
        watch(team1, team2, name1, name2, pick1, pick2, static, a.seed + i, a.clauses, a.turn_limit)


if __name__ == "__main__":
    main()
