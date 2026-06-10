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

from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM, TEAM_SIZE, featurize, stack_global
from cinnabar.engine_cpp import (  # inserts engine/build on sys.path
    Reveal, StaticData, build_state, final_material, load_teams, register_encoder, reveal_move)
import cinnabar_engine as ce  # noqa: E402
from cinnabar.policy import (  # noqa: E402
    MaxDamagePolicy, RandomPolicy, SmartHeuristicPolicy)
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
# Per-battle team source: returns one engine TeamSpec. Default picks a fixed team from the pool;
# --random-movesets swaps in the moveset sampler (set in main()).
_PICK_TEAM = lambda: random.choice(TEAMS)  # noqa: E731
_K = 1  # frame-stack depth (1 = off); set from --frame-stack in main(). Net global dim = GLOBAL_DIM*_K.
_CLAUSES = False  # OU Sleep+Freeze Clause on every training battle; set from --clauses in main().


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
                   help="self/league: fraction of iterations played vs the anchor (guards against drift)")
    p.add_argument("--anchor", choices=["smart", "maxdamage"], default="smart",
                   help="self/league: which fixed baseline to anchor a fraction of iters against")
    p.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"),
                   help="dir of Showdown team .txt files (a random one per side per battle)")
    p.add_argument("--random-movesets", action="store_true",
                   help="re-sample each species' moves per battle (keeps the pool's rosters); makes "
                        "moveset hidden info real and forces generalization (cinnabar/movesets.py)")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="stack the last K turns' global observations (1 = off); a cheap recent-history "
                        "probe before a full recurrent net. Net global dim becomes GLOBAL_DIM*K.")
    p.add_argument("--gen-teams", action="store_true",
                   help="generate a fresh team from the whole metagame per battle (any 6 viable-OU "
                        "species + sampled movesets) instead of the fixed pool. The scale-up environment.")
    p.add_argument("--clauses", action="store_true",
                   help="enable OU Sleep + Freeze Clause on every battle (matches the real format): a "
                        "second foe-inflicted sleep/freeze fails. Teaches correct status play; without "
                        "it the agent over-values sleep (it can sleep your whole team in training).")
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

def _pad(states, device, histories=None):
    """Featurize a list of states into padded tensors. Returns (glob, act, mask, feats).
    With frame-stacking (_K>1) and per-state `histories`, the global is the last _K globals
    concatenated (histories are updated in place); the stacked global is what gets stored."""
    raw = [featurize(s) for s in states]
    if _K > 1 and histories is not None:
        feats = [(stack_global(histories[i], g, _K), af) for i, (g, af) in enumerate(raw)]
    else:
        feats = raw
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


def select_batch(net, states, device, *, sample, record_buf=None, tags=None, histories=None) -> list[int]:
    """Choose an action for every state in one forward. Returns chosen indices (into each
    state's available_actions). If record_buf is given, append a StepRecord per state."""
    glob, act, mask, feats = _pad(states, device, histories)
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


def _select_opp(opp, states, device, histories=None) -> list[int]:
    if isinstance(opp, ActionScorer):
        return select_batch(opp, states, device, sample=True, histories=histories)
    return [opp.select_action(s).index for s in states]  # RandomPolicy / MaxDamagePolicy


# ----- PPO update (batched) -----------------------------------------------------------------

def _tensorize(steps, returns, device):
    import numpy as np

    n = len(steps)
    g_dim = len(steps[0].global_feats)
    a_dim = len(steps[0].action_feats[0])
    k = max(len(s.action_feats) for s in steps)
    # Build in numpy then convert once — feats from the fast path already ARE float32
    # numpy rows, so the per-row work is a memcpy, not a Python-float conversion.
    glob_np = np.zeros((n, g_dim), dtype=np.float32)
    act_np = np.zeros((n, k, a_dim), dtype=np.float32)
    mask_np = np.zeros((n, k), dtype=bool)
    for i, s in enumerate(steps):
        glob_np[i] = s.global_feats
        m = len(s.action_feats)
        act_np[i, :m] = s.action_feats
        mask_np[i, :m] = True
    glob = torch.from_numpy(glob_np)
    act = torch.from_numpy(act_np)
    mask = torch.from_numpy(mask_np)
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
        t1, t2 = _PICK_TEAM(), _PICK_TEAM()
        s1 = [(s, list(m)) for s, m in t1]
        s2 = [(s, list(m)) for s, m in t2]
        # r1/r2: persistent per-battle reveal sets (partial info + the agent's revealed-team
        # memory) for the Python path; o1/o2 are their C++ twins for the fast path.
        b = ce.make_battle(s1, s2, base + i)
        if _CLAUSES:
            b.set_clauses(True)  # OU Sleep+Freeze Clause for this battle
        items.append({"b": b, "s1": s1, "s2": s2,
                      "tag": f"{base}_{i}", "r1": Reveal(), "r2": Reveal(),
                      "o1": ce.Observer(), "o2": ce.Observer(),
                      "h0": [], "h1": []})  # per-side global-frame history (frame-stacking)
    return items


_STATIC: StaticData | None = None  # set in main(); the poke-env static-data cache for build_state


def _forward_select(net, glob, act, mask, device, *, sample, extras):
    """One batched forward straight over encode_batch's numpy outputs. Returns
    (chosen indices, behaviour log-probs or None, values or None)."""
    g, a = torch.from_numpy(glob), torch.from_numpy(act)
    m = torch.from_numpy(mask)
    if device != "cpu":
        g, a, m = g.to(device), a.to(device), m.to(device)
    with torch.no_grad():
        logits = net.score_actions_batch(g, a, m)
        logp_all = torch.log_softmax(logits, dim=1)
        chosen = (torch.multinomial(logp_all.exp(), 1).squeeze(1) if sample
                  else logits.argmax(dim=1))
        if not extras:
            return chosen.tolist(), None, None
        beh = logp_all.gather(1, chosen.unsqueeze(1)).squeeze(1).tolist()
        vals = net.value(g).tolist()
    return chosen.tolist(), beh, vals


def _run_fast(items, net, opp, args, *, record_buf, greedy_learner=False):
    """The C++-encoder rollout loop: per turn, ONE encode_batch call per side (features
    computed in C++, returned as numpy) + one batched forward, then step_pair (reveal
    bookkeeping + step, also C++). Same semantics as _run_py — the encoder itself is
    parity-tested bit-for-bit (tests/test_encoder_parity.py); this loop just avoids
    building Python BattleStates, which profiling showed was ~80% of rollout time."""
    if not ce.encoder_ready():
        register_encoder(_STATIC)
    opp_kind = None
    if not isinstance(opp, ActionScorer):  # map heuristic pilots to their C++ twins
        from cinnabar.search import _heuristic_kind  # one mapping for search + training
        opp_kind = _heuristic_kind(opp)
    for it in items:  # tolerate items built without observers (e.g. eval_ckpt's)
        it.setdefault("o1", ce.Observer())
        it.setdefault("o2", ce.Observer())
    for _ in range(args.turn_limit):
        live = [it for it in items if it["b"].result() == ce.Result.Ongoing]
        if not live:
            break
        battles = [it["b"] for it in live]
        glob, act, mask, mat, omat = ce.encode_batch(
            battles, [0] * len(live), [it["o1"] for it in live])
        idx1, beh, vals = _forward_select(net, glob, act, mask, args.device,
                                          sample=not greedy_learner,
                                          extras=record_buf is not None)
        if record_buf is not None:
            for j, it in enumerate(live):
                n = int(mask[j].sum())
                record_buf.setdefault(it["tag"], []).append(StepRecord(
                    glob[j].copy(), act[j, :n].copy(), idx1[j], beh[j], vals[j],
                    float(mat[j]), float(omat[j])))
        if isinstance(opp, ActionScorer):
            g2, a2, m2, _, _ = ce.encode_batch(
                battles, [1] * len(live), [it["o2"] for it in live])
            idx2, _, _ = _forward_select(opp, g2, a2, m2, args.device, sample=True, extras=False)
        elif opp_kind is not None:
            idx2 = [ce.select_heuristic(it["b"], 1, it["o2"], opp_kind) for it in live]
        else:  # RandomPolicy: uniform over the legal choices, no state needed
            idx2 = [random.randrange(len(it["b"].choices(1))) for it in live]
        for j, it in enumerate(live):
            ce.step_pair(it["b"], idx1[j], idx2[j], it["o1"], it["o2"])


def _run(items, net, opp, args, *, record_buf, greedy_learner=False):
    """Step all battles to completion, choosing actions in batched forwards each turn.
    Dispatches to the C++-encoder fast path; frame-stacking (_K > 1) keeps the Python
    featurization path, which is where the frame histories live."""
    if _K == 1:
        return _run_fast(items, net, opp, args, record_buf=record_buf,
                         greedy_learner=greedy_learner)
    return _run_py(items, net, opp, args, record_buf=record_buf, greedy_learner=greedy_learner)


def _run_py(items, net, opp, args, *, record_buf, greedy_learner=False):
    """The original Python-featurization loop (kept for frame-stacking and as a reference)."""
    for _ in range(args.turn_limit):
        live = [it for it in items if it["b"].result() == ce.Result.Ongoing]
        if not live:
            break
        st1 = [build_state(it["b"], 0, it["s1"], _STATIC, it["tag"], reveal=it["r1"], opp_team=it["s2"])
               for it in live]
        st2 = [build_state(it["b"], 1, it["s2"], _STATIC, it["tag"], reveal=it["r2"], opp_team=it["s1"])
               for it in live]
        a1 = select_batch(net, st1, args.device, sample=not greedy_learner,
                          record_buf=record_buf, tags=[it["tag"] for it in live] if record_buf is not None else None,
                          histories=[it["h0"] for it in live])
        a2 = _select_opp(opp, st2, args.device, histories=[it["h1"] for it in live])
        for j, it in enumerate(live):
            reveal_move(it["r1"], st2[j], st2[j].available_actions[a2[j]])  # p0 sees p1's move
            reveal_move(it["r2"], st1[j], st1[j].available_actions[a1[j]])  # p1 sees p0's move
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
    s = ActionScorer(GLOBAL_DIM * _K, ACTION_DIM, args.hidden).to(args.device)
    s.load_state_dict(net.state_dict())
    for p in s.parameters():
        p.requires_grad_(False)
    return s


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    global _STATIC, TEAMS, _PICK_TEAM, _K, _CLAUSES
    _K = max(1, args.frame_stack)
    _CLAUSES = bool(args.clauses)
    if _CLAUSES:
        print("clauses: ON (Sleep + Freeze Clause — second foe-inflicted sleep/freeze fails)")
    if _K > 1:
        print(f"frame-stack: {_K} (net global dim = {GLOBAL_DIM} x {_K} = {GLOBAL_DIM * _K})")
    _STATIC = StaticData(1)  # poke-env static data used by build_state
    register_encoder(_STATIC)  # upload the same tables into the C++ encoder (fast path)
    loaded = load_teams(args.teams_dir)
    if loaded:
        TEAMS = loaded
    print(f"teams: {len(TEAMS)} ({'from ' + args.teams_dir if loaded else 'fallback'})")
    if args.gen_teams:
        from cinnabar import movesets
        _PICK_TEAM = movesets.generate_team  # whole-metagame team generation (scale-up)
        print(f"gen-teams: ON ({len(movesets.ALL_SPECIES)} species, big-four weighted)")
    elif args.random_movesets:
        from cinnabar import movesets
        rosters = movesets.rosters_from_teams(TEAMS)
        _PICK_TEAM = lambda: movesets.sample_team(random.choice(rosters))  # noqa: E731
        print(f"random movesets: ON ({len(rosters)} rosters, per-species movepools)")

    net = ActionScorer(GLOBAL_DIM * _K, ACTION_DIM, args.hidden).to(args.device)
    if args.init:
        sd = torch.load(args.init, map_location=args.device)
        if (sd["policy_mlp.0.weight"].shape[1] != GLOBAL_DIM * _K + ACTION_DIM
                or sd["value_mlp.0.weight"].shape[1] != GLOBAL_DIM * _K):  # older checkpoint -> auto-pad
            from pad_checkpoint import pad_state_dict
            og = sd["value_mlp.0.weight"].shape[1]
            pad_state_dict(sd, _K)
            print(f"init: auto-padded G {og} -> {GLOBAL_DIM * _K} + A {ACTION_DIM} (new features zeroed, fair warm-start)")
        net.load_state_dict(sd)
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
    # self/league: anchor a fraction of iters against a fixed baseline so self-play can't drift
    # off into beating only its past selves. Smart is the stronger, more informative anchor.
    anchor = SmartHeuristicPolicy() if args.anchor == "smart" else MaxDamagePolicy()
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
