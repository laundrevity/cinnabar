"""Phase 2 training: REINFORCE-with-baseline, sparse win/loss reward.

Algorithm (kept deliberately simple — the goal is "does it learn at all"):

  1. Play a batch of battles: our PGPolicy (sampling) vs a fixed opponent
     (MaxDamagePolicy by default). Each turn's (features, chosen action) is
     recorded, grouped by battle.
  2. When a battle ends, its only reward is +1 (win) / -1 (loss). We spread that
     terminal reward back over the battle's moves with a discount (returns.py).
  3. Standardize returns across the batch, then do one policy-gradient step:
        policy loss  = -sum( logp(action) * (return - V(state)) )
        value loss   =  sum( (V(state) - return)^2 )
        + an entropy bonus to keep exploring.
  4. Periodically evaluate greedily vs Random and MaxDamage, and checkpoint.

Needs a local Showdown server (../scripts/run-server.sh). Training runs real
battles, so it's not fast — use --smoke first to confirm the loop runs.

    cd agent
    uv run python train.py --smoke          # tiny run, just checks it works
    uv run python train.py                   # real run (default hyperparameters)
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch

from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM
from cinnabar.policy import MaxDamagePolicy, RandomPolicy
from cinnabar.rl.agent import PGPolicy
from cinnabar.rl.net import ActionScorer
from cinnabar.rl.returns import discounted_returns, standardize
from cinnabar.showdown import PolicyPlayer

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM = (REPO_ROOT / "teams" / "gen1ou-sample.txt").read_text()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the RL agent (REINFORCE + baseline).")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--batch", type=int, default=30, help="battles collected per update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--tie-reward", type=float, default=-1.0,
                   help="reward for a turn-limit draw; -1 treats a stall as a loss")
    p.add_argument("--step-penalty", type=float, default=0.0,
                   help="per-turn penalty to discourage stalling (try 0.01 if stalls persist)")
    p.add_argument("--opponent", choices=["random", "maxdamage"], default="random",
                   help="training opponent (curriculum: beat random first, then maxdamage)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-battles", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--out", default="models")
    p.add_argument("--init", default=None, help="checkpoint to load weights from (for curriculum)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the loop works")
    args = p.parse_args()
    if args.smoke:
        args.iters, args.batch, args.eval_every, args.eval_battles, args.ckpt_every = 2, 4, 1, 4, 2
    return args


def update(net, optimizer, steps, returns, value_coef, ent_coef, device) -> dict:
    rets = torch.tensor(standardize(returns), dtype=torch.float32, device=device)
    optimizer.zero_grad()

    policy_loss = torch.zeros((), device=device)
    value_loss = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)

    for rec, ret in zip(steps, rets):
        g = torch.tensor(rec.global_feats, dtype=torch.float32, device=device)
        a = torch.tensor(rec.action_feats, dtype=torch.float32, device=device)
        logits = net.score_actions(g, a)
        dist = torch.distributions.Categorical(logits=logits)
        value = net.value(g)
        advantage = ret - value.detach()
        policy_loss = policy_loss - dist.log_prob(torch.tensor(rec.chosen, device=device)) * advantage
        value_loss = value_loss + (value - ret) ** 2
        entropy = entropy + dist.entropy()

    n = max(len(steps), 1)
    loss = (policy_loss + value_coef * value_loss - ent_coef * entropy) / n
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item() / n,
        "value_loss": value_loss.item() / n,
        "entropy": entropy.item() / n,
    }


def make_player(policy, concurrency):
    return PolicyPlayer(
        policy=policy, battle_format="gen1ou", team=TEAM, max_concurrent_battles=concurrency
    )


async def eval_winrate(policy: PGPolicy, learner: PolicyPlayer, opponent: PolicyPlayer, n: int) -> float:
    policy.eval()
    learner.reset_battles()
    await learner.battle_against(opponent, n_battles=n)
    policy.train()
    return 100.0 * learner.n_won_battles / max(learner.n_finished_battles, 1)


async def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    net = ActionScorer(GLOBAL_DIM, ACTION_DIM, args.hidden).to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device))
        print(f"Initialized weights from {args.init}")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = PGPolicy(net, device=args.device)

    learner = make_player(policy, args.concurrency)
    maxdmg = make_player(MaxDamagePolicy(), args.concurrency)
    rng = make_player(RandomPolicy(), args.concurrency)
    train_opp = maxdmg if args.opponent == "maxdamage" else rng

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.iters} iters x {args.batch} battles vs {args.opponent}. Saving to {out}/")

    for it in range(1, args.iters + 1):
        policy.train()
        policy.reset_buffer()
        learner.reset_battles()
        await learner.battle_against(train_opp, n_battles=args.batch)

        steps, returns, wins, ties, total = [], [], 0, 0, 0
        for tag, recs in policy.steps_by_battle.items():
            battle = learner.battles.get(tag)
            if battle is None or not battle.finished or not recs:
                continue  # only skip genuinely unfinished battles
            total += 1
            if battle.won is True:
                wins += 1
                reward = 1.0
            elif battle.won is False:
                reward = -1.0
            else:  # turn-limit draw — a non-win, not a free pass
                ties += 1
                reward = args.tie_reward
            rewards = [-args.step_penalty] * len(recs)
            rewards[-1] += reward
            returns.extend(discounted_returns(rewards, args.gamma))
            steps.extend(recs)

        info = update(net, optimizer, steps, returns, args.value_coef, args.ent_coef, args.device) if steps else {"loss": float("nan")}
        train_wr = 100.0 * wins / max(total, 1)
        print(f"iter {it:4d} | train vs {args.opponent:9s} {train_wr:5.1f}% | ties {ties:2d} | loss {info['loss']:+.3f} | steps {len(steps)}")

        if it % args.eval_every == 0:
            wr_rng = await eval_winrate(policy, learner, rng, args.eval_battles)
            wr_md = await eval_winrate(policy, learner, maxdmg, args.eval_battles)
            print(f"         eval (greedy) | vs random {wr_rng:5.1f}% | vs maxdmg {wr_md:5.1f}%")

        if it % args.ckpt_every == 0 or it == args.iters:
            path = out / f"pg_iter{it}.pt"
            torch.save(net.state_dict(), path)
            print(f"         saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
