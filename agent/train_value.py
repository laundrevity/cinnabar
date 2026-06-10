"""Train a CALIBRATED win-probability value head — the search leaf the agent has been missing.

Why: the PPO value head is a shaped-return baseline, not a probability (measured MSE ~0.48 vs
win/loss — worse than always saying 0.5). Search only uses its *rankings*, which is why 1-ply works
at all, but a leaf that can't price tempo, incapacitation ("my sleeping Rhydon contributes
nothing") or lost matchups can't see past next turn's HP bars — the exact failure a strong human
exploits (browser ground-truth game, 2026-06-09).

What: supervised regression P(win) ~ game outcome (BCEWithLogits) on states from a DELIBERATELY
diverse distribution:
  * mixed pilots (net greedy/sampled, smart, maxdamage, staller, random) — positions self-play
    never reaches,
  * mixed teams (the teams/ pool + whole-metagame generated teams),
  * `--afflict`: random pre-game status injection via the engine's state-injection API, so
    sleeping/frozen/paralyzed-mon positions are OVERSAMPLED instead of vanishingly rare,
  * both players' perspectives every turn, clauses on.

The result drops into search via HybridNet (policy net proposes, this leaf disposes) — no search
code changes: `search_eval.py --value-ckpt ...`, `play.py --search --value-ckpt ...`.

    cd agent
    uv run python train_value.py --ckpt models_cf/pg_best.pt --games 3000 --out value_net
    uv run python search_eval.py --ckpt models_cf/pg_best.pt --value-ckpt value_net/value_best.pt \
        --battles 100 --clauses --top-k 3 --opponent staller
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cinnabar import movesets
from cinnabar.encoding import GLOBAL_DIM, encode_global
from cinnabar.engine_cpp import Reveal, StaticData, build_state, load_teams, reveal_move
from cinnabar.policy import MaxDamagePolicy, RandomPolicy, SmartHeuristicPolicy, StallerPolicy
from cinnabar.rl.net import ValueNet
from cinnabar.search import search_action_values_minimax
from ladder import _load_net, _load_value

import cinnabar_engine as ce  # noqa: E402

STATUSES = ["SLP", "PAR", "FRZ", "BRN", "PSN"]


class EngineSearchPilot:
    """A data-generation pilot that plays by the DEPLOYED search configuration (minimax + gating +
    null-prune, p=0.5 from the knob sweep). Value iteration: the leaf's training games are played
    by the agent the leaf will serve — strong-play positions (freeze-fishing duels, win-condition
    management) finally exist in the data. ~50x slower than a policy pilot; use via --search-frac."""

    def __init__(self, net, static, top_k=3, opp_top_k=0, rollouts=2, paranoia=0.5, device="cpu"):
        self.net, self.static = net, static
        self.top_k, self.opp_top_k = top_k, opp_top_k
        self.rollouts, self.paranoia, self.device = rollouts, paranoia, device

    def pick(self, battle, player, state, my_spec, opp_spec):
        cands, vals = search_action_values_minimax(
            battle, player, self.net, self.static, my_spec, opp_spec,
            reveal=None, device=self.device, rollouts=self.rollouts, state=state,
            top_k=self.top_k, opp_top_k=self.opp_top_k, paranoia=self.paranoia)
        return state.available_actions[cands[max(range(len(vals)), key=vals.__getitem__)]]


def play_collect(p1, p2, team1, team2, static, seed, clauses, turn_limit, rng,
                 afflict: float) -> list[tuple[list[float], float]]:
    """One game; returns (global_features, outcome-for-that-player) for BOTH players, every turn.
    With prob `afflict`, a random non-active status is injected per side pre-game (state-injection
    API) so incapacitated positions exist in the data at scale."""
    spec1 = [(s, list(m)) for s, m in team1]
    spec2 = [(s, list(m)) for s, m in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    for player in (0, 1):
        if rng.random() < afflict:
            slot = rng.randrange(len(spec1 if player == 0 else spec2))
            st = rng.choice(STATUSES)
            battle.set_mon_state(player, slot, rng.uniform(0.3, 1.0), status=st,
                                 sleep_turns=rng.randint(1, 5) if st == "SLP" else 0)
    r1, r2 = Reveal(), Reveal()
    feats: list[tuple[int, list[float]]] = []
    turns = 0

    def choose(p, player, state, my, opp):
        if isinstance(p, EngineSearchPilot):
            return p.pick(battle, player, state, my, opp)
        return p.select_action(state)

    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, "tv", reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, "tv_o", reveal=r2, opp_team=spec1)
        feats.append((0, encode_global(s1)))
        feats.append((1, encode_global(s2)))
        a1 = choose(p1, 0, s1, spec1, spec2)
        a2 = choose(p2, 1, s2, spec2, spec1)
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    res = battle.result()
    out0 = 0.5 if res in (ce.Result.Tie, ce.Result.Ongoing) else (1.0 if res == ce.Result.P1Win else 0.0)
    return [(g, out0 if pl == 0 else 1.0 - out0) for pl, g in feats]


def calibration_report(probs: torch.Tensor, ys: torch.Tensor) -> None:
    brier = float(((probs - ys) ** 2).mean())
    print(f"  holdout Brier {brier:.4f}  (always-0.5 = 0.25; the OLD head measured ~0.48 as MSE)")
    print(f"  {'bucket':>12s} {'n':>7s} {'pred':>6s} {'actual':>7s}")
    for lo in [i / 10 for i in range(10)]:
        m = (probs >= lo) & (probs < lo + 0.1)
        if int(m.sum()) == 0:
            continue
        print(f"  [{lo:.1f}, {lo + 0.1:.1f}) {int(m.sum()):7d} {float(probs[m].mean()):6.2f} "
              f"{float(ys[m].mean()):7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a calibrated win-prob value head (search leaf).")
    ap.add_argument("--ckpt", required=True, help="policy checkpoint (net pilots for data diversity)")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--afflict", type=float, default=0.35,
                    help="per-side prob of a random pre-game status injection (incapacitation coverage)")
    ap.add_argument("--search-frac", type=float, default=0.0,
                    help="prob a side is piloted by the DEPLOYED search config (minimax p=0.5, "
                         "gated, null-pruned) — value iteration; ~50x slower per search game")
    ap.add_argument("--search-value-ckpt", default=None,
                    help="leaf for the search PILOT (e.g. the previous value_best.pt — iteration "
                         "k trains on games judged by iteration k-1's leaf). Default: PPO head.")
    ap.add_argument("--gen-frac", type=float, default=0.5, help="fraction of games on generated teams")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-clauses", dest="clauses", action="store_false")
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--out", default="value_net")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(clauses=True)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    static = StaticData(1)
    teams = load_teams(a.teams_dir)

    net_pol = _load_net(a.ckpt, a.hidden, a.device, 1)   # greedy
    sampled = _load_net(a.ckpt, a.hidden, a.device, 1)   # the same net, sampled (exploration)
    try:
        from cinnabar.rl.agent import PGPolicy
        sampled = PGPolicy(net_pol.net, device=a.device)
        sampled.record = False  # sample actions, never train
    except Exception:
        pass
    pilots = [net_pol, net_pol, sampled, SmartHeuristicPolicy(), SmartHeuristicPolicy(),
              MaxDamagePolicy(), StallerPolicy(), RandomPolicy(seed=a.seed)]
    searcher = None
    if a.search_frac > 0:
        search_net = net_pol.net
        if a.search_value_ckpt:
            from cinnabar.rl.net import HybridNet
            search_net = HybridNet(net_pol.net, _load_value(a.search_value_ckpt, a.hidden, a.device))
        searcher = EngineSearchPilot(search_net, static, device=a.device)

    def pick_pilot():
        if searcher is not None and rng.random() < a.search_frac:
            return searcher
        return rng.choice(pilots)

    def pick_team():
        if rng.random() < a.gen_frac:
            return [(s, list(m)) for s, m in movesets.generate_team(rng)]
        return rng.choice(teams)

    print(f"collecting: {a.games} games, afflict {a.afflict}, gen-frac {a.gen_frac}, "
          f"search-frac {a.search_frac}, clauses {'on' if a.clauses else 'off'}")
    data: list[tuple[list[float], float]] = []
    for i in range(a.games):
        p1, p2 = pick_pilot(), pick_pilot()
        data += play_collect(p1, p2, pick_team(), pick_team(), static, 1000 + i,
                             a.clauses, a.turn_limit, rng, a.afflict)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} games, {len(data)} states")
    rng.shuffle(data)
    n_hold = int(len(data) * a.holdout)
    hold, train = data[:n_hold], data[n_hold:]
    print(f"{len(train)} train / {len(hold)} holdout states")

    X = torch.tensor([g for g, _ in train], dtype=torch.float32, device=a.device)
    Y = torch.tensor([y for _, y in train], dtype=torch.float32, device=a.device)
    Xh = torch.tensor([g for g, _ in hold], dtype=torch.float32, device=a.device)
    Yh = torch.tensor([y for _, y in hold], dtype=torch.float32, device=a.device)

    vnet = ValueNet(GLOBAL_DIM, a.hidden).to(a.device)
    opt = torch.optim.Adam(vnet.parameters(), lr=a.lr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(a.epochs):
        vnet.train()
        perm = torch.randperm(len(X), device=a.device)
        tot, nb = 0.0, 0
        for i in range(0, len(X), a.batch):
            idx = perm[i:i + a.batch]
            loss = F.binary_cross_entropy_with_logits(vnet(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        vnet.eval()
        with torch.no_grad():
            hold_bce = float(F.binary_cross_entropy_with_logits(vnet(Xh), Yh))
        flag = ""
        if hold_bce < best:
            best = hold_bce
            torch.save(vnet.state_dict(), out / "value_best.pt")
            flag = "  -> value_best.pt"
        print(f"epoch {ep}: train BCE {tot / max(nb, 1):.4f}  holdout BCE {hold_bce:.4f}{flag}")

    vnet.load_state_dict(torch.load(out / "value_best.pt", map_location=a.device))
    vnet.eval()
    with torch.no_grad():
        calibration_report(vnet.value(Xh), Yh)
    print(f"\ndone -> {out}/value_best.pt   (plug in via --value-ckpt on search_eval / play.py)")


if __name__ == "__main__":
    main()
