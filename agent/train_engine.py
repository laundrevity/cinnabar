"""Engine-backed training: PPO on the fast in-process C++ engine (no Showdown server).

Same algorithm as train.py (per-action scorer, clipped PPO, sparse/shaped/dense reward),
but battles run synchronously via engine_cpp — so it trains at engine speed, not network
speed. The learner is p1 and records its trajectory; the opponent is p2 (random / maxdamage
/ a frozen self snapshot).

    cd agent
    uv run python train_engine.py --smoke        # tiny run, checks the loop
    uv run python train_engine.py                  # real run

v1 caveats: full-information observations, and a fixed pool of teams using only the moves
the engine fully models (see TEAMS).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM, TEAM_SIZE
from cinnabar.engine_cpp import StaticData, final_material, play_battle  # inserts engine/build on sys.path
import cinnabar_engine as ce  # noqa: E402
from cinnabar.policy import MaxDamagePolicy, RandomPolicy  # noqa: E402
from cinnabar.rl.agent import PGPolicy  # noqa: E402
from cinnabar.rl.net import ActionScorer  # noqa: E402
from cinnabar.rl.returns import discounted_returns, standardize  # noqa: E402

# Engine-clean teams (only fully-modeled moves). A battle draws two at random.
TEAMS = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the RL agent on the C++ engine (PPO).")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch", type=int, default=64, help="battles per update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--tie-reward", type=float, default=-1.0)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--reward", choices=["sparse", "shaped", "dense"], default="shaped")
    p.add_argument("--faint-value", type=float, default=0.5)
    p.add_argument("--dmg-value", type=float, default=1.0)
    p.add_argument("--opponent", choices=["random", "maxdamage", "self"], default="self")
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-battles", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--out", default="models_engine")
    p.add_argument("--init", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--turn-limit", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.batch, args.eval_every, args.eval_battles, args.ckpt_every = 2, 8, 1, 8, 2
    return args


def ppo_update(net, optimizer, steps, returns, *, epochs, minibatch_size, clip,
               value_coef, ent_coef, device) -> dict:
    values_old = [s.value for s in steps]
    advantages = standardize([r - v for r, v in zip(returns, values_old)])
    order = list(range(len(steps)))
    info = {"loss": float("nan")}
    for _ in range(epochs):
        random.shuffle(order)
        for start in range(0, len(order), minibatch_size):
            mb = order[start:start + minibatch_size]
            optimizer.zero_grad()
            policy_loss = torch.zeros((), device=device)
            value_loss = torch.zeros((), device=device)
            entropy = torch.zeros((), device=device)
            for i in mb:
                rec = steps[i]
                g = torch.tensor(rec.global_feats, dtype=torch.float32, device=device)
                a = torch.tensor(rec.action_feats, dtype=torch.float32, device=device)
                dist = torch.distributions.Categorical(logits=net.score_actions(g, a))
                logp = dist.log_prob(torch.tensor(rec.chosen, device=device))
                value = net.value(g)
                ratio = torch.exp(logp - rec.behavior_logp)
                adv = advantages[i]
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
                policy_loss = policy_loss - torch.min(ratio * adv, clipped)
                value_loss = value_loss + (value - returns[i]) ** 2
                entropy = entropy + dist.entropy()
            m = max(len(mb), 1)
            loss = (policy_loss + value_coef * value_loss - ent_coef * entropy) / m
            loss.backward()
            optimizer.step()
            info = {"loss": loss.item(), "policy_loss": policy_loss.item() / m,
                    "value_loss": value_loss.item() / m, "entropy": entropy.item() / m}
    return info


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


def rollout(learner, opp, static, args, base) -> tuple[list, list, int, int, int]:
    learner.train()
    learner.reset_buffer()
    battles = []
    for i in range(args.batch):
        t1, t2 = random.choice(TEAMS), random.choice(TEAMS)
        tag = f"{base}_{i}"
        battles.append((tag, play_battle(learner, opp, t1, t2, static,
                                         seed=base * 100003 + i, tag=tag, turn_limit=args.turn_limit)))
    steps, returns, wins, ties, total = [], [], 0, 0, 0
    for tag, battle in battles:
        recs = learner.steps_by_battle.get(tag)
        if not recs:
            continue
        total += 1
        res = battle.result()
        if res == ce.Result.P1Win:
            wins += 1
        elif res != ce.Result.P2Win:
            ties += 1
        returns.extend(discounted_returns(build_rewards(recs, battle, args), args.gamma))
        steps.extend(recs)
    return steps, returns, wins, ties, total


def eval_winrate(learner, opp, static, n, base) -> float:
    learner.eval()
    wins = 0
    for i in range(n):
        t1, t2 = random.choice(TEAMS), random.choice(TEAMS)
        b = play_battle(learner, opp, t1, t2, static, seed=base * 7919 + i, tag=f"ev{i}")
        if b.result() == ce.Result.P1Win:
            wins += 1
    learner.train()
    return 100.0 * wins / max(n, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    static = StaticData(1)

    net = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    learner = PGPolicy(net, device=args.device)

    if args.opponent == "self":
        opp_net = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
        opp_net.load_state_dict(net.state_dict())
        opp = PGPolicy(opp_net, device=args.device)
        opp.record = False  # samples for diversity, never trains
    elif args.opponent == "maxdamage":
        opp, opp_net = MaxDamagePolicy(), None
    else:
        opp, opp_net = RandomPolicy(), None

    rng_eval, md_eval = RandomPolicy(), MaxDamagePolicy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Engine PPO: {args.iters} iters x {args.batch} battles vs {args.opponent} "
          f"[{args.reward} reward]. -> {out}/")

    best_md = -1.0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        steps, returns, wins, ties, total = rollout(learner, opp, static, args, it)
        info = (ppo_update(net, optimizer, steps, returns, epochs=args.epochs,
                           minibatch_size=args.minibatch_size, clip=args.clip,
                           value_coef=args.value_coef, ent_coef=args.ent_coef, device=args.device)
                if steps else {"loss": float("nan")})
        bps = args.batch / max(time.time() - t0, 1e-9)
        print(f"iter {it:4d} | win {100.0 * wins / max(total, 1):5.1f}% | ties {ties:2d} | "
              f"loss {info['loss']:+.3f} | steps {len(steps):5d} | {bps:5.1f} battles/s")

        if it % args.eval_every == 0:
            wr_rng = eval_winrate(learner, rng_eval, static, args.eval_battles, it)
            wr_md = eval_winrate(learner, md_eval, static, args.eval_battles, it + 1)
            print(f"         eval (greedy) | vs random {wr_rng:5.1f}% | vs maxdmg {wr_md:5.1f}%")
            if wr_md > best_md:
                best_md = wr_md
                torch.save(net.state_dict(), out / "pg_best.pt")

        if opp_net is not None and it % args.snapshot_every == 0:
            opp_net.load_state_dict(net.state_dict())  # opponent catches up
        if it % args.ckpt_every == 0 or it == args.iters:
            torch.save(net.state_dict(), out / f"pg_iter{it}.pt")


if __name__ == "__main__":
    main()
