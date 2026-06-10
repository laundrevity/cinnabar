"""Elo ladder — a strength metric that doesn't saturate.

Win% vs any single fixed baseline tops out (the agent beats random/maxdamage/smart all
~70-100%), so it stops measuring progress. A ladder plays a round-robin among the fixed
baselines AND a set of checkpoints, then fits Bradley-Terry / Elo ratings: everyone is ranked
relative to everyone, so improvement past the heuristic shows up as Elo, and self-play
snapshots can be ordered against each other.

    cd agent
    uv run python ladder.py --ckpts models_obs2_smart/pg_best.pt
    uv run python ladder.py --ckpts models_league/pg_iter*.pt --battles 150 --mirror

Each pair plays `--battles` games, split evenly across which side leads (cancels any P1 edge);
`--mirror` gives both sides the same team (removes team-matchup luck, isolating skill).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

import train_engine as T  # select_batch (greedy net action), _STATIC machinery
from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
from cinnabar.engine_cpp import Reveal, StaticData, build_state, load_teams, reveal_move
import cinnabar_engine as ce  # noqa: E402
from cinnabar.policy import (MaxDamagePolicy, Policy, RandomPolicy, SmartHeuristicPolicy,
                             StallerPolicy)
from cinnabar.rl.net import ActionScorer
from cinnabar.search import search_action_index, search_action_values_minimax
from cinnabar.state import BattleState


class NetPolicy(Policy):
    """Wrap a trained ActionScorer as a greedy Policy (argmax over legal actions). With frame-stack
    (k>1) it keeps a per-battle global history, reset when the battle tag changes."""

    def __init__(self, net, device: str = "cpu", k: int = 1) -> None:
        self.net, self.device, self.k = net, device, k
        self._hist: list[list[float]] = []
        self._tag = None

    def select_action(self, state: BattleState):
        if self.k > 1:
            if state.battle_tag != self._tag:   # new battle -> fresh history
                self._hist, self._tag = [], state.battle_tag
            idx = T.select_batch(self.net, [state], self.device, sample=False, histories=[self._hist])[0]
        else:
            idx = T.select_batch(self.net, [state], self.device, sample=False)[0]
        return state.available_actions[idx]


def _load_net(path: str, hidden: int, device: str, k: int = 1) -> NetPolicy:
    net = ActionScorer(GLOBAL_DIM * k, ACTION_DIM, hidden).to(device)
    sd = torch.load(path, map_location=device)
    if (sd["policy_mlp.0.weight"].shape[1] != GLOBAL_DIM * k + ACTION_DIM
            or sd["value_mlp.0.weight"].shape[1] != GLOBAL_DIM * k):  # older/smaller checkpoint -> auto-pad
        from pad_checkpoint import pad_state_dict
        og, oa = sd["value_mlp.0.weight"].shape[1], sd["policy_mlp.0.weight"].shape[1]
        pad_state_dict(sd, k)
        print(f"  (auto-padded {Path(path).name}: G {og}->{GLOBAL_DIM * k}, policy width {oa}->"
              f"{GLOBAL_DIM * k + ACTION_DIM}; new features zeroed)")
    net.load_state_dict(sd)
    net.eval()
    return NetPolicy(net, device, k)


def _load_value(path: str, hidden: int, device: str):
    """Load a calibrated win-prob ValueNet (train_value.py) — the drop-in search leaf."""
    from cinnabar.rl.net import ValueNet

    vnet = ValueNet(GLOBAL_DIM, hidden).to(device)
    vnet.load_state_dict(torch.load(path, map_location=device))
    vnet.eval()
    return vnet


class PolicyPilot:
    """A ladder seat driven by a Policy (heuristics, NetPolicy): picks from the BattleState."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def choose(self, battle, side, state, reveal, my_spec, opp_spec, static, rng) -> int:
        return self.policy.select_action(state).index


class SearchPilot:
    """A ladder seat driven by decision-time search on the live battle — the configuration
    that actually plays in the browser (play.py). Rating it on the same Elo scale as the
    raw checkpoints measures what lookahead is worth, not just that it helps vs one fixed
    opponent (search_eval's job)."""

    def __init__(self, net, *, rollouts=3, top_k=3, minimax=False, opp_top_k=0,
                 opp_temp=0.0, device="cpu") -> None:
        self.net, self.device = net, device
        self.rollouts, self.top_k = rollouts, top_k
        self.minimax, self.opp_top_k, self.opp_temp = minimax, opp_top_k, opp_temp
        self.opp_model = SmartHeuristicPolicy()  # the assumed opponent inside the lookahead

    def choose(self, battle, side, state, reveal, my_spec, opp_spec, static, rng) -> int:
        if self.minimax:
            cands, vals = search_action_values_minimax(
                battle, side, self.net, static, my_spec, opp_spec, reveal=reveal,
                device=self.device, rollouts=self.rollouts, rng=rng, state=state,
                top_k=self.top_k, opp_top_k=self.opp_top_k, opp_temp=self.opp_temp)
            return cands[max(range(len(vals)), key=vals.__getitem__)]
        return search_action_index(
            battle, side, self.net, self.opp_model, static, my_spec, opp_spec, reveal=reveal,
            device=self.device, rollouts=self.rollouts, rng=rng, state=state, top_k=self.top_k)


def _play_one(pa, pb, t1, t2, static, seed, turn_limit, clauses, rng):
    """One battle between two pilots (pa as P1). Mirrors play_search_battle's loop: both
    sides keep a Reveal memory, see each other's selected moves, and step together."""
    spec1 = [(s, list(m)) for s, m in t1]
    spec2 = [(s, list(m)) for s, m in t2]
    b = ce.make_battle(spec1, spec2, seed)
    if clauses:
        b.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    turns = 0
    while b.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(b, 0, spec1, static, "lad", reveal=r1, opp_team=spec2)
        s2 = build_state(b, 1, spec2, static, "lad_o", reveal=r2, opp_team=spec1)
        i1 = pa.choose(b, 0, s1, r1, spec1, spec2, static, rng)
        i2 = pb.choose(b, 1, s2, r2, spec2, spec1, static, rng)
        a1, a2 = s1.available_actions[i1], s2.available_actions[i2]
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        b.step(b.choices(0)[a1.index], b.choices(1)[a2.index])
    return b.result()


def _play(p1, p2, teams, n, base, mirror, static, turn_limit, pick_team=None, clauses=False):
    """Return p1's score (win=1, tie=0.5) over n games, split across both lead positions."""
    pick_team = pick_team or (lambda: random.choice(teams))
    rng = random.Random(base * 977 + 13)  # search dice (policy pilots ignore it)
    score = 0.0
    for i in range(n):
        t1 = pick_team()
        t2 = t1 if mirror else pick_team()
        a, b = (p1, p2) if i % 2 == 0 else (p2, p1)  # alternate who leads (cancels P1 edge)
        r = _play_one(a, b, t1, t2, static, base + i, turn_limit, clauses, rng)
        p1_is_a = i % 2 == 0
        if r == ce.Result.Tie or r == ce.Result.Ongoing:
            score += 0.5
        elif (r == ce.Result.P1Win) == p1_is_a:  # the side p1 played on won
            score += 1.0
    return score


def _elo(names, wins, games, anchor="random"):
    """Bradley-Terry MLE (Hunter MM), Laplace-smoothed, reported on the Elo scale."""
    n = len(names)
    p = [1.0] * n
    sw = [[wins[i][j] + 0.5 for j in range(n)] for i in range(n)]   # +0.5 smoothing
    sg = [[games[i][j] + 1.0 for j in range(n)] for i in range(n)]
    for _ in range(1000):
        new = p[:]
        for i in range(n):
            num = sum(sw[i][j] for j in range(n) if j != i)
            den = sum(sg[i][j] / (p[i] + p[j]) for j in range(n) if j != i)
            new[i] = num / den if den else p[i]
        gm = math.exp(sum(math.log(max(x, 1e-9)) for x in new) / n)  # normalise geo-mean to 1
        new = [x / gm for x in new]
        if max(abs(a - b) for a, b in zip(p, new)) < 1e-9:
            p = new
            break
        p = new
    elo = [400.0 * math.log10(max(x, 1e-9)) for x in p]
    shift = 1000.0 - (elo[names.index(anchor)] if anchor in names else min(elo))
    return [e + shift for e in elo]


def main() -> None:
    ap = argparse.ArgumentParser(description="Elo ladder over baselines + checkpoints.")
    ap.add_argument("--ckpts", nargs="*", default=[], help="checkpoint .pt files to rate")
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--battles", type=int, default=120, help="games per pair")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--mirror", action="store_true", help="same team both sides (isolate skill)")
    ap.add_argument("--random-movesets", action="store_true",
                    help="re-sample movesets per battle (match training; cinnabar/movesets.py)")
    ap.add_argument("--gen-teams", action="store_true",
                    help="generate whole teams from the metagame per battle (match --gen-teams training)")
    ap.add_argument("--frame-stack", type=int, default=1, help="stack last K turns' globals (match training)")
    ap.add_argument("--clauses", action="store_true",
                    help="enable OU Sleep + Freeze Clause (match --clauses training / the real format)")
    ap.add_argument("--search-ckpts", nargs="*", default=[],
                    help="checkpoints to rate as SEARCH pilots (decision-time lookahead, the "
                         "browser configuration) — same Elo scale as the raw checkpoints")
    ap.add_argument("--search-rollouts", type=int, default=3)
    ap.add_argument("--search-top-k", type=int, default=3)
    ap.add_argument("--search-minimax", action="store_true",
                    help="search pilots use adversarial lookahead (with --search-opp-top-k/temp)")
    ap.add_argument("--search-opp-top-k", type=int, default=3)
    ap.add_argument("--search-opp-temp", type=float, default=0.0)
    ap.add_argument("--no-staller", action="store_true",
                    help="drop the staller baseline (it slows pairs down a little)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    T._K = max(1, a.frame_stack)  # _pad reads train_engine._K for frame-stacking; set it here too

    static = StaticData(1)
    teams = load_teams(a.teams_dir) or T._FALLBACK_TEAMS
    pick_team = None
    if a.gen_teams:
        from cinnabar import movesets
        pick_team = movesets.generate_team  # whole-metagame teams (scale-up)
    elif a.random_movesets:
        from cinnabar import movesets
        rosters = movesets.rosters_from_teams(teams)
        pick_team = lambda: movesets.sample_team(random.choice(rosters))  # noqa: E731

    players: list[tuple[str, object]] = [
        ("random", PolicyPilot(RandomPolicy())), ("maxdamage", PolicyPilot(MaxDamagePolicy())),
        ("smart", PolicyPilot(SmartHeuristicPolicy())),
    ]
    if not a.no_staller:
        players.append(("staller", PolicyPilot(StallerPolicy())))
    for c in a.ckpts:
        players.append((Path(c).parent.name + "/" + Path(c).stem,
                        PolicyPilot(_load_net(c, a.hidden, a.device, max(1, a.frame_stack)))))
    for c in a.search_ckpts:
        tag = "mm" if a.search_minimax else f"k{a.search_top_k}"
        players.append((Path(c).parent.name + "/" + Path(c).stem + f"+search[{tag}]",
                        SearchPilot(_load_net(c, a.hidden, a.device, 1).net,
                                    rollouts=a.search_rollouts, top_k=a.search_top_k,
                                    minimax=a.search_minimax, opp_top_k=a.search_opp_top_k,
                                    opp_temp=a.search_opp_temp, device=a.device)))

    names = [n for n, _ in players]
    k = len(players)
    wins = [[0.0] * k for _ in range(k)]
    games = [[0.0] * k for _ in range(k)]
    base = 1
    for i in range(k):
        for j in range(i + 1, k):
            s = _play(players[i][1], players[j][1], teams, a.battles, base, a.mirror, static, a.turn_limit,
                      pick_team=pick_team, clauses=a.clauses)
            base += a.battles
            wins[i][j], wins[j][i] = s, a.battles - s
            games[i][j] = games[j][i] = a.battles

    elo = _elo(names, wins, games)
    order = sorted(range(k), key=lambda i: -elo[i])
    print(f"\nElo ladder  ({a.battles} games/pair{', mirror' if a.mirror else ''}, anchor random=1000)\n")
    print(f"  {'player':28s} {'elo':>6s}  {'win%':>6s}")
    for i in order:
        tot_w = sum(wins[i][j] for j in range(k) if j != i)
        tot_g = sum(games[i][j] for j in range(k) if j != i)
        print(f"  {names[i]:28s} {elo[i]:6.0f}  {100.0 * tot_w / max(tot_g, 1):6.1f}")


if __name__ == "__main__":
    main()
