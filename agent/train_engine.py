"""Engine-backed training: PPO on the fast in-process C++ engine (no Showdown server).

Same algorithm as train.py (per-action scorer, clipped PPO, sparse/shaped/dense reward),
but fully vectorized for throughput:

  * rollouts run all `batch` battles concurrently — every battle's action is chosen in ONE
    batched forward per turn (learner and self-opponent), not one battle/step at a time;
  * the PPO update runs each minibatch as one padded/masked forward+backward.

    cd agent
    uv run python train_engine.py --smoke        # tiny run, checks the loop
    uv run python train_engine.py                  # real run

v1 caveats: full-information observations, and a fixed pool of teams using only moves the
engine fully models (see TEAMS).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM, TEAM_SIZE, featurize
from cinnabar.engine_cpp import StaticData, build_state, final_material, load_teams  # inserts engine/build on sys.path
import cinnabar_engine as ce  # noqa: E402
from cinnabar.policy import MaxDamagePolicy, RandomPolicy, SmartHeuristicPolicy  # noqa: E402
from cinnabar.rl.agent import StepRecord  # noqa: E402
from cinnabar.rl.net import ActionScorer  # noqa: E402
from cinnabar.rl.returns import discounted_returns, standardize  # noqa: E402

# Fallback teams (only fully-modeled moves); --teams-dir loads the teams/ pool in main().
_FALLBACK_TEAMS = [
    [("Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]),
     ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
     ("Exeggutor", ["Psychic", "Sleep Powder", "Explosion", "Body Slam"]),
     ("Starmie", ["Thunderbolt", "Ice Beam", "Recover", "Thunder Wave"]),
     ("Chansey", ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
     ("Alakazam", ["Psychic", "Thunder Wave", "Recover", "Seismic Toss"])],
    [("Zapdos", ["Thunderbolt", "Drill Peck", "Thunder Wave", "Rest"]),
     ("Rhydon", ["Earthquake", "Body Slam", "Blizzard", "Fire Blast"]),
     ("Lapras", ["Blizzard", "Surf", "Body Slam", "Rest"]),
     ("Gengar", ["Psychic", "Thunderbolt", "Explosion", "Thunder Wave"]),
     ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
     ("Jynx", ["Blizzard", "Psychic", "Body Slam", "Seismic Toss"])],
]
TEAMS = _FALLBACK_TEAMS  # set from --teams-dir in main() if any teams load


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the RL agent on the C++ engine (vectorized PPO).")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch", type=int, default=256, help="concurrent battles per update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--tie-reward", type=float, default=-1.0)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--reward", choices=["sparse", "shaped", "dense"], default="shaped")
    p.add_argument("--faint-value", type=float, default=0.5)
    p.add_argument("--dmg-value", type=float, default=1.0)
    p.add_argument("--opponent", choices=["random", "maxdamage", "smart", "self", "league"],
                   default="self")
    p.add_argument("--snapshot-every", type=int, default=10,
                   help="self: refresh the opponent every N iters; league: add a snapshot every N iters")
    p.add_argument("--anchor-frac", type=float, default=0.5,
                   help="self/league: fraction of iterations played vs max-damage (anchors against drift)")
    p.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"),
                   help="dir of Showdown team .txt files (a random one per side per battle)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-battles", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--out", default="models_engine")
    p.add_argument("--init", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--turn-limit", type=int, default=500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.batch, args.eval_every, args.eval_battles, args.ckpt_every = 2, 16, 1, 16, 2
    return args


# ----- batched featurization / action selection --------------------------------------------

def _pad(states, device):
    """Featurize a list of states into padded tensors. Returns (glob, act, mask, feats)."""
    feats = [featurize(s) for s in states]
    b = len(feats)
    g_dim = len(feats[0][0])
    a_dim = len(feats[0][1][0])
    k = max(len(af) for _, af in feats)
    glob = torch.zeros(b, g_dim)
    act = torch.zeros(b, k, a_dim)
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i, (g, af) in enumerate(feats):
        glob[i] = torch.tensor(g, dtype=torch.float32)
        n = len(af)
        act[i, :n] = torch.tensor(af, dtype=torch.float32)
        mask[i, :n] = True
    return glob.to(device), act.to(device), mask.to(device), feats


def select_batch(net, states, device, *, sample, record_buf=None, tags=None) -> list[int]:
    """Choose an action for every state in one forward. Returns chosen indices (into each
    state's available_actions). If record_buf is given, append a StepRecord per state."""
    glob, act, mask, feats = _pad(states, device)
    with torch.no_grad():
        logits = net.score_actions_batch(glob, act, mask)
        logp_all = torch.log_softmax(logits, dim=1)
        chosen = (torch.multinomial(logp_all.exp(), 1).squeeze(1) if sample
                  else logits.argmax(dim=1))
        chosen_l = chosen.tolist()
        if record_buf is not None:
            beh = logp_all.gather(1, chosen.unsqueeze(1)).squeeze(1).tolist()
            vals = net.value(glob).tolist()
    if record_buf is not None:
        for i, s in enumerate(states):
            our = sum(m.hp_fraction for m in s.team)
            opp = sum(m.hp_fraction for m in s.opponent_team)  # full info (v1)
            record_buf.setdefault(tags[i], []).append(
                StepRecord(feats[i][0], feats[i][1], chosen_l[i], beh[i], vals[i], our, opp))
    return chosen_l


def _select_opp(opp, states, device) -> list[int]:
    if isinstance(opp, ActionScorer):
        return select_batch(opp, states, device, sample=True)
    return [opp.select_action(s).index for s in states]  # RandomPolicy / MaxDamagePolicy


# ----- PPO update (batched) -----------------------------------------------------------------

def _tensorize(steps, returns, device):
    n = len(steps)
    g_dim = len(steps[0].global_feats)
    a_dim = len(steps[0].action_feats[0])
    k = max(len(s.action_feats) for s in steps)
    glob = torch.zeros(n, g_dim)
    act = torch.zeros(n, k, a_dim)
    mask = torch.zeros(n, k, dtype=torch.bool)
    for i, s in enumerate(steps):
        glob[i] = torch.tensor(s.global_feats)
        m = len(s.action_feats)
        act[i, :m] = torch.tensor(s.action_feats)
        mask[i, :m] = True
    chosen = torch.tensor([s.chosen for s in steps], dtype=torch.long)
    beh = torch.tensor([s.behavior_logp for s in steps], dtype=torch.float32)
    ret = torch.tensor(returns, dtype=torch.float32)
    return (glob.to(device), act.to(device), mask.to(device),
            chosen.to(device), beh.to(device), ret.to(device))


def ppo_update(net, optimizer, steps, returns, *, epochs, minibatch_size, clip,
               value_coef, ent_coef, device) -> dict:
    glob, act, mask, chosen, beh, ret = _tensorize(steps, returns, device)
    adv = torch.tensor(standardize([r - s.value for r, s in zip(returns, steps)]),
                       dtype=torch.float32, device=device)
    n = len(steps)
    info = {"loss": float("nan")}
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            optimizer.zero_grad()
            logits = net.score_actions_batch(glob[idx], act[idx], mask[idx])
            logp_all = torch.log_softmax(logits, dim=1)
            logp = logp_all.gather(1, chosen[idx].unsqueeze(1)).squeeze(1)
            entropy = -(logp_all.exp() * logp_all.masked_fill(~mask[idx], 0.0)).sum(dim=1)
            value = net.value(glob[idx])
            ratio = torch.exp(logp - beh[idx])
            a = adv[idx]
            policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            value_loss = ((value - ret[idx]) ** 2).mean()
            ent = entropy.mean()
            loss = policy_loss + value_coef * value_loss - ent_coef * ent
            loss.backward()
            optimizer.step()
            info = {"loss": loss.item(), "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(), "entropy": ent.item()}
    return info


# ----- rewards / rollout / eval -------------------------------------------------------------

def build_rewards(recs, battle, args) -> list[float]:
    res = battle.result()
    win = 1.0 if res == ce.Result.P1Win else (-1.0 if res == ce.Result.P2Win else args.tie_reward)
    if args.reward == "dense":
        mats = [(r.our_material, r.opp_material) for r in recs] + [final_material(battle)]
        rewards = []
        for t in range(len(recs)):
            (o0, p0), (o1, p1) = mats[t], mats[t + 1]
            rewards.append(args.dmg_value * ((p0 - p1) - (o0 - o1)) - args.step_penalty)
        rewards[-1] += win
        return rewards
    base = win
    if args.reward == "shaped":
        our_f = sum(1 for e in battle.team_state(0) if e[3])
        opp_f = sum(1 for e in battle.team_state(1) if e[3])
        base += args.faint_value * (opp_f - our_f) / TEAM_SIZE
    rewards = [-args.step_penalty] * len(recs)
    rewards[-1] += base
    return rewards


def _make_battles(n, base):
    items = []
    for i in range(n):
        t1, t2 = random.choice(TEAMS), random.choice(TEAMS)
        s1 = [(s, list(m)) for s, m in t1]
        s2 = [(s, list(m)) for s, m in t2]
        items.append({"b": ce.make_battle(s1, s2, base + i), "s1": s1, "s2": s2, "tag": f"{base}_{i}"})
    return items


_STATIC: StaticData | None = None  # set in main(); the poke-env static-data cache for build_state


def _run(items, net, opp, args, *, record_buf, greedy_learner=False):
    """Step all battles to completion, choosing actions in batched forwards each turn."""
    for _ in range(args.turn_limit):
        live = [it for it in items if it["b"].result() == ce.Result.Ongoing]
        if not live:
            break
        st1 = [build_state(it["b"], 0, it["s1"], _STATIC, it["tag"]) for it in live]
        st2 = [build_state(it["b"], 1, it["s2"], _STATIC, it["tag"]) for it in live]
        a1 = select_batch(net, st1, args.device, sample=not greedy_learner,
                          record_buf=record_buf, tags=[it["tag"] for it in live] if record_buf is not None else None)
        a2 = _select_opp(opp, st2, args.device)
        for j, it in enumerate(live):
            it["b"].step(it["b"].choices(0)[a1[j]], it["b"].choices(1)[a2[j]])


def rollout(net, opp, args, base):
    items = _make_battles(args.batch, base * 100003)
    buf: dict = {}
    _run(items, net, opp, args, record_buf=buf)
    steps, returns, wins, ties, total = [], [], 0, 0, 0
    for it in items:
        recs = buf.get(it["tag"])
        if not recs:
            continue
        total += 1
        res = it["b"].result()
        if res == ce.Result.P1Win:
            wins += 1
        elif res != ce.Result.P2Win:
            ties += 1
        returns.extend(discounted_returns(build_rewards(recs, it["b"], args), args.gamma))
        steps.extend(recs)
    return steps, returns, wins, ties, total


def eval_winrate(net, opp, args, n, base) -> float:
    items = _make_battles(n, base * 7919)
    _run(items, net, opp, args, record_buf=None, greedy_learner=True)
    wins = sum(1 for it in items if it["b"].result() == ce.Result.P1Win)
    return 100.0 * wins / max(n, 1)


def _snapshot(net, args):
    """A frozen copy of the learner network (a self-play / league opponent)."""
    s = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
    s.load_state_dict(net.state_dict())
    for p in s.parameters():
        p.requires_grad_(False)
    return s


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    global _STATIC, TEAMS
    _STATIC = StaticData(1)  # poke-env static data used by build_state
    loaded = load_teams(args.teams_dir)
    if loaded:
        TEAMS = loaded
    print(f"teams: {len(TEAMS)} ({'from ' + args.teams_dir if loaded else 'fallback'})")

    net = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Opponent: a fixed policy/snapshot, or a growing league pool of past snapshots.
    pool = None
    if args.opponent == "league":
        pool = [_snapshot(net, args)]  # grows over training; each iter plays a random member
        fixed_opp = None
    elif args.opponent == "self":
        fixed_opp = _snapshot(net, args)  # one snapshot, refreshed every snapshot-every
    elif args.opponent == "maxdamage":
        fixed_opp = MaxDamagePolicy()
    elif args.opponent == "smart":
        fixed_opp = SmartHeuristicPolicy()
    else:
        fixed_opp = RandomPolicy()

    rng_eval, md_eval, sm_eval = RandomPolicy(), MaxDamagePolicy(), SmartHeuristicPolicy()
    anchor = MaxDamagePolicy()  # self/league: anchor a fraction of iters against this baseline
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Engine PPO (vectorized): {args.iters} iters x {args.batch} battles vs {args.opponent} "
          f"[{args.reward} reward]. -> {out}/")

    # Rank checkpoints on the SmartHeuristic yardstick — max-damage is too weak and exploitable
    # to rank by (mirror eval showed its win% is ~uncorrelated with real strength). Seeding from
    # the loaded net also stops a resume from clobbering a good pg_best.pt on the first eval.
    best_sm = -1.0
    if args.init:
        best_sm = eval_winrate(net, sm_eval, args, args.eval_battles, 0)
        print(f"init checkpoint: vs smart {best_sm:5.1f}% (pg_best.pt overwritten only if beaten)")

    for it in range(1, args.iters + 1):
        t0 = time.time()
        if args.opponent in ("self", "league") and random.random() < args.anchor_frac:
            opp = anchor  # anchor this iter vs max-damage so self-play can't drift off the benchmark
        elif pool is not None:
            opp = random.choice(pool)  # league: a random past self
        else:
            opp = fixed_opp
        steps, returns, wins, ties, total = rollout(net, opp, args, it)
        info = (ppo_update(net, optimizer, steps, returns, epochs=args.epochs,
                           minibatch_size=args.minibatch_size, clip=args.clip,
                           value_coef=args.value_coef, ent_coef=args.ent_coef, device=args.device)
                if steps else {"loss": float("nan")})
        bps = args.batch / max(time.time() - t0, 1e-9)
        print(f"iter {it:4d} | win {100.0 * wins / max(total, 1):5.1f}% | ties {ties:2d} | "
              f"loss {info['loss']:+.3f} | steps {len(steps):6d} | {bps:6.0f} battles/s")

        if it % args.eval_every == 0:
            wr_rng = eval_winrate(net, rng_eval, args, args.eval_battles, it)
            wr_md = eval_winrate(net, md_eval, args, args.eval_battles, it + 1)
            wr_sm = eval_winrate(net, sm_eval, args, args.eval_battles, it + 2)
            print(f"         eval (greedy) | vs random {wr_rng:5.1f}% | vs maxdmg {wr_md:5.1f}% "
                  f"| vs smart {wr_sm:5.1f}%")
            if wr_sm > best_sm:
                best_sm = wr_sm
                torch.save(net.state_dict(), out / "pg_best.pt")

        if it % args.snapshot_every == 0:
            if args.opponent == "self":
                fixed_opp.load_state_dict(net.state_dict())  # opponent catches up to the learner
            elif pool is not None:
                pool.append(_snapshot(net, args))  # add a new league member
                print(f"         league pool: {len(pool)} snapshots")
        if it % args.ckpt_every == 0 or it == args.iters:
            torch.save(net.state_dict(), out / f"pg_iter{it}.pt")


if __name__ == "__main__":
    main()
