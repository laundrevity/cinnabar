"""Expert iteration — distil decision-time search back into the policy (AlphaZero-lite).

Search plays ~19% above the raw policy, but pays that cost every move. Expert iteration folds the gain
back into the network so it's free at inference:

  Each round: self-play games where BOTH sides choose moves by search (strong play), recording every
  decision as (state, the action search chose, the eventual game outcome). Then train the policy to
  IMITATE the search action (cross-entropy) and the value head to predict the outcome (MSE). The
  improved net is what next round's search builds on, so play and evaluation compound.

No hand-coded strategy and no human teams — the only inputs are the game rules and win/loss. The
value head is re-targeted to win/loss, which also makes it a better-calibrated leaf for the search.
The clause-fail action feature (ACTION_DIM 23) is learned here for the first time, so the re-sleep
should fall out too.

    cd agent
    uv run python expert_iter.py --init models_clauses/pg_best.pt --rounds 5 --games 60 \
        --gen-teams --clauses --out ei
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cinnabar import movesets
from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM, featurize
from cinnabar.engine_cpp import Reveal, StaticData, build_state, load_teams, play_battle, reveal_move
from cinnabar.policy import SmartHeuristicPolicy
from cinnabar.search import search_action_index
from ladder import NetPolicy, _load_net

import cinnabar_engine as ce  # noqa: E402

Sample = tuple  # (global: list, actions: list[list], target: int, outcome: float)


def generate_game(net, opp_model, team1, team2, static, seed, device, rollouts, clauses, turn_limit):
    """One self-play game, both sides by search. Returns the agent decisions as training samples."""
    spec1 = [(s, list(m)) for s, m in team1]
    spec2 = [(s, list(m)) for s, m in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    recs: list[tuple] = []  # (player, global, actions, target)
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, "ei", reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, "ei_o", reveal=r2, opp_team=spec1)
        i1 = search_action_index(battle, 0, net, opp_model, static, spec1, spec2,
                                 reveal=r1, device=device, rollouts=rollouts)
        i2 = search_action_index(battle, 1, net, opp_model, static, spec2, spec1,
                                 reveal=r2, device=device, rollouts=rollouts)
        if len(s1.available_actions) > 1:  # only record real choices
            g, a = featurize(s1)
            recs.append((0, g, a, i1))
        if len(s2.available_actions) > 1:
            g, a = featurize(s2)
            recs.append((1, g, a, i2))
        a1, a2 = s1.available_actions[i1], s2.available_actions[i2]
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    res = battle.result()
    out0 = 0.5 if res in (ce.Result.Tie, ce.Result.Ongoing) else (1.0 if res == ce.Result.P1Win else 0.0)
    return [(g, a, t, out0 if pl == 0 else 1.0 - out0) for (pl, g, a, t) in recs]


def train(net, opt, samples, device, epochs, batch_size, value_coef):
    net.train()
    for ep in range(epochs):
        random.shuffle(samples)
        tot = pol = val = 0.0
        nb = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            b = len(batch)
            k = max(len(a) for _, a, _, _ in batch)
            gf = torch.zeros(b, GLOBAL_DIM)
            af = torch.zeros(b, k, ACTION_DIM)
            mask = torch.zeros(b, k, dtype=torch.bool)
            tgt = torch.zeros(b, dtype=torch.long)
            out = torch.zeros(b)
            for j, (g, a, t, o) in enumerate(batch):
                gf[j] = torch.tensor(g)
                for m, av in enumerate(a):
                    af[j, m] = torch.tensor(av)
                    mask[j, m] = True
                tgt[j], out[j] = t, o
            gf, af, mask, tgt, out = (x.to(device) for x in (gf, af, mask, tgt, out))
            logits = net.score_actions_batch(gf, af, mask)
            p_loss = F.cross_entropy(logits, tgt)
            v_loss = F.mse_loss(net.value(gf), out)
            loss = p_loss + value_coef * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); pol += p_loss.item(); val += v_loss.item(); nb += 1
        nb = max(nb, 1)
        print(f"    epoch {ep}: loss {tot/nb:.4f}  (policy {pol/nb:.4f}, value {val/nb:.4f})")
    net.eval()


def quick_eval(net, device, teams, static, clauses, n=120, seed0=77):
    """Raw greedy win% vs the smart heuristic — tracks whether the DISTILLED policy is improving."""
    pol = NetPolicy(net, device, 1)
    smart = SmartHeuristicPolicy()
    w = 0.0
    for i in range(n):
        t1, t2 = random.choice(teams), random.choice(teams)
        lead = i % 2 == 0
        a, b = (pol, smart) if lead else (smart, pol)
        r = play_battle(a, b, t1, t2, static, seed0 + i, tag=f"ev{seed0+i}", clauses=clauses).result()
        if r in (ce.Result.Tie, ce.Result.Ongoing):
            w += 0.5
        elif (r == ce.Result.P1Win) == lead:
            w += 1.0
    return w / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Expert iteration: distil search into the policy.")
    ap.add_argument("--init", required=True, help="starting checkpoint (auto-padded to current dims)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--games", type=int, default=60, help="self-play games per round")
    ap.add_argument("--rollouts", type=int, default=3, help="search rollouts per action")
    ap.add_argument("--epochs", type=int, default=4, help="distillation epochs per round")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--gen-teams", action="store_true", help="whole-metagame teams (else the teams/ pool)")
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--out", default="ei")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)

    net = _load_net(a.init, a.hidden, a.device, 1).net  # auto-pads (incl. the new clause feature)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    opp_model = SmartHeuristicPolicy()  # the opponent the search assumes

    teams = load_teams(a.teams_dir)
    rng = random.Random(a.seed)
    pick = (lambda: movesets.generate_team(rng)) if a.gen_teams else (lambda: random.choice(teams))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"expert iteration: rounds={a.rounds} games/round={a.games} rollouts={a.rollouts} "
          f"clauses={'on' if a.clauses else 'off'}  teams={'gen' if a.gen_teams else len(teams)}")
    base = 1
    print(f"  init: raw greedy vs smart {quick_eval(net, a.device, teams, static, a.clauses)*100:.1f}%")
    for r in range(a.rounds):
        samples: list = []
        for _ in range(a.games):
            t1, t2 = pick(), pick()
            samples += generate_game(net, opp_model, t1, t2, static, base, a.device,
                                     a.rollouts, a.clauses, a.turn_limit)
            base += 1
        print(f"\nround {r}: {len(samples)} decisions from {a.games} search self-play games")
        train(net, opt, samples, a.device, a.epochs, a.batch, a.value_coef)
        torch.save(net.state_dict(), out / f"ei_round{r}.pt")
        torch.save(net.state_dict(), out / "pg_best.pt")
        print(f"  round {r}: raw greedy vs smart "
              f"{quick_eval(net, a.device, teams, static, a.clauses)*100:.1f}%  -> {out}/pg_best.pt")

    print(f"\nexpert iteration done -> {out}/pg_best.pt")


if __name__ == "__main__":
    main()
