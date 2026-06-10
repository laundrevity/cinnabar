"""Expert iteration v2 — distil decision-time search into the policy, with the v1 failure fixed.

v1 (commit fd0f049) DEGRADED the policy (62.9% -> 40% vs smart) despite search being a genuinely
better target. Three measured causes, each fixed here:

  1. **Opponent-model mismatch.** v1's search assumed a *heuristic* opponent while actually facing
     *search* in self-play — mis-calibrated targets. Now the lookahead models the opponent as the
     CURRENT POLICY (greedy), which both tracks the distilled net as it improves and is far closer
     to what the opponent actually plays (`--opp-model heuristic` restores the old behaviour).
  2. **Hard CE to the argmax.** v1 imitated only search's single chosen action — a target the net
     can't represent (CE plateaued ~1.3), grinding the policy into an incoherent greedy. Now the
     target is the DISTRIBUTION softmax(Q/tau) over the searched candidates (AlphaZero-style soft
     targets): the net learns "these actions are comparably good" instead of a brittle argmax.
  3. **No anchor — catastrophic forgetting.** v1's pure imitation on a narrow self-play
     distribution overwrote a coherent PPO policy. Now a trust-region KL term to the FROZEN initial
     policy keeps the distillation local (`--anchor-coef`), and `pg_best.pt` is only written when a
     round actually beats the running best on eval (vs smart AND vs the staller) — the v1 run left
     a broken net in pg_best.

  Plus the lever that didn't exist in v1: the teacher is **policy-prior gated search** (`--top-k`,
  default 3) — measured +20.8% over raw vs the staller and stronger than ungated search by +15.8%.
  The policy proposes, the value head disposes, and the distillation folds the result back in.

No hand-coded strategy and no human teams — inputs are the game rules and win/loss. The value head
is re-targeted to win/loss, which also makes it a better-calibrated leaf for next round's search.

    cd agent
    uv run python expert_iter.py --init models_wf/pg_best.pt --rounds 5 --games 60 \
        --gen-teams --clauses --out ei2
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cinnabar import movesets
from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM, featurize
from cinnabar.engine_cpp import Reveal, StaticData, build_state, load_teams, play_battle, reveal_move
from cinnabar.policy import SmartHeuristicPolicy, StallerPolicy
from cinnabar.search import play_search_battle, search_action_values
from ladder import NetPolicy, _load_net

import cinnabar_engine as ce  # noqa: E402

Sample = tuple  # (global, action_feats, candidates, q_values, outcome)


def generate_game(net, opp_model, team1, team2, static, seed, device, rollouts, clauses,
                  turn_limit, top_k, opp_policy=None):
    """One teacher game. Both sides by gated search (self-play), or — when `opp_policy` is given —
    P2 plays that policy directly (an ANCHOR game: keeps non-mirror styles, e.g. relentless attack,
    in the distillation data; train_engine's --anchor-frac trick). Records (state, candidate
    Q-vector, outcome) per real decision — the full search distribution, not just its argmax.
    Anchor games record only P1's (search) decisions."""
    spec1 = [(s, list(m)) for s, m in team1]
    spec2 = [(s, list(m)) for s, m in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    recs: list[tuple] = []  # (player, global, action_feats, candidates, values)
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, "ei", reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, "ei_o", reveal=r2, opp_team=spec1)
        c1, v1 = search_action_values(battle, 0, net, opp_model, static, spec1, spec2,
                                      reveal=r1, device=device, rollouts=rollouts,
                                      state=s1, top_k=top_k)
        if len(s1.available_actions) > 1:
            g, a = featurize(s1)
            recs.append((0, g, a, c1, v1))
        i1 = c1[max(range(len(v1)), key=v1.__getitem__)]
        if opp_policy is not None:
            i2 = opp_policy.select_action(s2).index
        else:
            c2, v2 = search_action_values(battle, 1, net, opp_model, static, spec2, spec1,
                                          reveal=r2, device=device, rollouts=rollouts,
                                          state=s2, top_k=top_k)
            if len(s2.available_actions) > 1:
                g, a = featurize(s2)
                recs.append((1, g, a, c2, v2))
            i2 = c2[max(range(len(v2)), key=v2.__getitem__)]
        a1, a2 = s1.available_actions[i1], s2.available_actions[i2]
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    res = battle.result()
    out0 = 0.5 if res in (ce.Result.Tie, ce.Result.Ongoing) else (1.0 if res == ce.Result.P1Win else 0.0)
    return [(g, a, c, v, out0 if pl == 0 else 1.0 - out0) for (pl, g, a, c, v) in recs]


def train(net, init_net, opt, samples, device, epochs, batch_size, value_coef, anchor_coef, tau,
          margin=0.0):
    """Distil: soft-CE to softmax(Q/tau) over searched candidates + value MSE to outcome
    + KL(init || current) trust region so the PPO policy isn't overwritten wholesale.

    `margin > 0` switches to DECISIVE-ONLY distillation: train (hard CE) only on samples where the
    teacher overruled the policy's argmax (candidates[0], by topk order) by a Q-gap >= margin, and
    apply no policy gradient elsewhere. Rationale (measured): Q-spreads over a gated top-3 are
    usually near-ties, so soft targets ~= uniform over top-3 — distilling them just injects entropy
    and FLATTENS the policy (raw greedy dropped ~6-10 pts at round 0 in every v2.x run while
    teacher-match stayed ~50%, i.e. the soft objective was already optimized). The teacher's edge
    lives in the sparse decisive disagreements; the loss should be equally sparse."""
    net.train()
    if margin > 0.0:
        gaps = sorted(max(q) - q[0] for _, _, _, q, _ in samples
                      if max(range(len(q)), key=q.__getitem__) != 0)
        kept = [s for s in samples if max(range(len(s[3])), key=s[3].__getitem__) != 0
                and max(s[3]) - s[3][0] >= margin]
        if gaps:
            pct = [gaps[int(len(gaps) * p)] for p in (0.5, 0.75, 0.9)]
            print(f"    decisive filter: {len(gaps)}/{len(samples)} disagreements "
                  f"(gap p50/p75/p90 = {pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}); "
                  f"kept {len(kept)} with gap >= {margin}")
        samples = kept
        if not samples:
            print("    decisive filter: nothing to train on this round")
            net.eval()
            return
    for ep in range(epochs):
        random.shuffle(samples)
        tot = pol = val = anc = match = 0.0
        nb = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            b = len(batch)
            k = max(len(a) for _, a, _, _, _ in batch)
            gf = torch.zeros(b, GLOBAL_DIM)
            af = torch.zeros(b, k, ACTION_DIM)
            mask = torch.zeros(b, k, dtype=torch.bool)
            target = torch.zeros(b, k)
            out = torch.zeros(b)
            for j, (g, a, cand, q, o) in enumerate(batch):
                gf[j] = torch.tensor(g)
                for m, av in enumerate(a):
                    af[j, m] = torch.tensor(av)
                    mask[j, m] = True
                if margin > 0.0:  # decisive-only: hard target on the teacher's overruling pick
                    target[j, cand[max(range(len(q)), key=q.__getitem__)]] = 1.0
                else:  # soft target: softmax over the searched candidates' Q-values; zero elsewhere
                    qt = torch.tensor(q)
                    probs = torch.softmax((qt - qt.max()) / tau, dim=0)
                    for ci, p in zip(cand, probs):
                        target[j, ci] = p
                out[j] = o
            gf, af, mask, target, out = (x.to(device) for x in (gf, af, mask, target, out))

            logits = net.score_actions_batch(gf, af, mask)        # -inf at padded slots
            logp = F.log_softmax(logits, dim=1)
            # target is 0 at padded/-inf positions: mask the product so 0 * -inf can't make NaNs.
            p_loss = -torch.where(mask, target * logp, torch.zeros_like(logp)).sum(1).mean()
            v_loss = F.mse_loss(net.value(gf), out)
            with torch.no_grad():
                init_logp = F.log_softmax(init_net.score_actions_batch(gf, af, mask), dim=1)
                init_p = init_logp.exp()
                # Teacher-match: does the student's argmax equal the teacher's? Separates
                # "underfit" (low match) from "no headroom" (high match, flat eval).
                match += (logits.argmax(1) == target.argmax(1)).float().mean().item()
            kl = torch.where(mask, init_p * (init_logp - logp), torch.zeros_like(logp)).sum(1).mean()
            loss = p_loss + value_coef * v_loss + anchor_coef * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            pol += p_loss.item()
            val += v_loss.item()
            anc += kl.item()
            nb += 1
        nb = max(nb, 1)
        print(f"    epoch {ep}: loss {tot/nb:.4f}  (policy {pol/nb:.4f}, value {val/nb:.4f}, "
              f"anchor-KL {anc/nb:.4f}, teacher-match {match/nb*100:.0f}%)")
    net.eval()


def quick_eval(net, device, teams, static, clauses, opponent, n=300, seed0=77):
    """Raw greedy win% vs `opponent` — tracks whether the DISTILLED policy itself improves.
    Team picks come from a PRIVATE rng seeded by seed0, so every round evaluates the identical
    matchup schedule (the global rng is advanced by game generation — using it made round evals
    unpaired: a round-0 'drop' was measured with zero gradient steps taken)."""
    pol = NetPolicy(net, device, 1)
    pick_t = random.Random(seed0)
    w = 0.0
    for i in range(n):
        t1, t2 = pick_t.choice(teams), pick_t.choice(teams)
        lead = i % 2 == 0
        pa, pb = (pol, opponent) if lead else (opponent, pol)
        r = play_battle(pa, pb, t1, t2, static, seed0 + i, tag=f"ev{seed0+i}", clauses=clauses).result()
        if r in (ce.Result.Tie, ce.Result.Ongoing):
            w += 0.5
        elif (r == ce.Result.P1Win) == lead:
            w += 1.0
    return w / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Expert iteration v2: distil gated search into the policy.")
    ap.add_argument("--init", required=True, help="starting checkpoint (auto-padded to current dims)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--games", type=int, default=60, help="self-play games per round")
    ap.add_argument("--rollouts", type=int, default=3, help="search rollouts per action")
    ap.add_argument("--top-k", type=int, default=3,
                    help="policy-prior gating for the search teacher (0 = search all actions)")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="soft-target temperature over candidate Q-values (value-head units, ~[0,1])")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="decisive-only distillation: train (hard CE) ONLY where the teacher "
                         "overrules the policy argmax by a Q-gap >= margin; 0 = soft targets on "
                         "everything (measured to flatten the policy). Round 0 prints the gap "
                         "percentiles — pick the margin from those.")
    ap.add_argument("--anchor-coef", type=float, default=0.5,
                    help="KL(initial policy || current) trust-region weight (0 = v1's free-fall)")
    ap.add_argument("--opp-model", choices=["policy", "heuristic"], default="policy",
                    help="the opponent the lookahead assumes (policy = matched-ish to self-play)")
    ap.add_argument("--anchor-frac", type=float, default=0.34,
                    help="fraction of teacher games played vs the smart heuristic instead of "
                         "search-mirror (keeps non-mirror styles in the data; v2.0's pure-mirror "
                         "data caused a staller-shaped regression). 0 = pure self-play")
    ap.add_argument("--eval-battles", type=int, default=300,
                    help="games per eval line (120 had SE~4.5%% — too noisy to gate keep-best)")
    ap.add_argument("--eval-search", type=int, default=0,
                    help="ALSO measure gated-search strength per round with N games per opponent "
                         "(the AlphaZero criterion: the PLAYER improves through a better value head "
                         "and prior even when raw greedy doesn't; slow). Included in keep-best.")
    ap.add_argument("--epochs", type=int, default=4, help="distillation epochs per round")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--gen-teams", action="store_true", help="whole-metagame teams (else the teams/ pool)")
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--out", default="ei2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)

    net = _load_net(a.init, a.hidden, a.device, 1).net  # auto-pads to current dims
    init_net = copy.deepcopy(net).eval()                # the frozen trust-region anchor
    for p in init_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    opp_model = NetPolicy(net, a.device, 1) if a.opp_model == "policy" else SmartHeuristicPolicy()

    teams = load_teams(a.teams_dir)
    rng = random.Random(a.seed)
    pick = (lambda: movesets.generate_team(rng)) if a.gen_teams else (lambda: random.choice(teams))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smart, staller = SmartHeuristicPolicy(), StallerPolicy()

    def search_strength(opponent, n, seed0):
        """Gated-search (current net as prior + value head) win% vs `opponent` as P1."""
        w = 0.0
        pick_t = random.Random(seed0)
        for i in range(n):
            t1, t2 = pick_t.choice(teams), pick_t.choice(teams)
            r = play_search_battle(net, opponent, opp_model, t1, t2, static, seed0 + i,
                                   clauses=a.clauses, device=a.device, rollouts=a.rollouts,
                                   top_k=a.top_k).result()
            if r in (ce.Result.Tie, ce.Result.Ongoing):
                w += 0.5
            elif r == ce.Result.P1Win:
                w += 1.0
        return w / n

    def full_eval(tag):
        ws = quick_eval(net, a.device, teams, static, a.clauses, smart, n=a.eval_battles)
        wt = quick_eval(net, a.device, teams, static, a.clauses, staller, n=a.eval_battles, seed0=5077)
        line = f"  {tag}: raw greedy vs smart {ws*100:.1f}%  vs staller {wt*100:.1f}%"
        scores = [ws, wt]
        if a.eval_search > 0:
            ss = search_strength(smart, a.eval_search, 9090)
            st = search_strength(staller, a.eval_search, 9690)
            line += f"  |  SEARCH vs smart {ss*100:.1f}%  vs staller {st*100:.1f}%"
            scores += [ss, st]
        print(line)
        return sum(scores) / len(scores)

    print(f"expert iteration v2: rounds={a.rounds} games/round={a.games} rollouts={a.rollouts} "
          f"top-k={a.top_k} tau={a.tau} anchor={a.anchor_coef} opp-model={a.opp_model} "
          f"anchor-frac={a.anchor_frac} clauses={'on' if a.clauses else 'off'}  "
          f"teams={'gen' if a.gen_teams else len(teams)}")
    best = full_eval("init")
    base = 1
    for r in range(a.rounds):
        samples: list = []
        n_anchor = 0
        for gi in range(a.games):
            t1, t2 = pick(), pick()
            anchor_game = (gi % max(round(1 / a.anchor_frac), 1) == 0) if a.anchor_frac > 0 else False
            n_anchor += 1 if anchor_game else 0
            samples += generate_game(net, opp_model, t1, t2, static, base, a.device,
                                     a.rollouts, a.clauses, a.turn_limit, a.top_k,
                                     opp_policy=smart if anchor_game else None)
            base += 1
        print(f"\nround {r}: {len(samples)} decisions from {a.games} gated-search games "
              f"({n_anchor} vs the smart anchor, rest self-play)")
        train(net, init_net, opt, samples, a.device, a.epochs, a.batch, a.value_coef,
              a.anchor_coef, a.tau, margin=a.margin)
        torch.save(net.state_dict(), out / f"ei_round{r}.pt")
        score = full_eval(f"round {r}")
        if score > best:  # pg_best only on a real improvement (the v1 run left a broken pg_best)
            best = score
            torch.save(net.state_dict(), out / "pg_best.pt")
            print(f"    new best ({best*100:.1f}% mean) -> {out}/pg_best.pt")

    if not (out / "pg_best.pt").exists():
        print(f"\nno round beat the init checkpoint — {out}/pg_best.pt not written (init remains best)")
    print(f"\nexpert iteration done. round ckpts in {out}/")


if __name__ == "__main__":
    main()
