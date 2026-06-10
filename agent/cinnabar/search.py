"""Decision-time search: shallow lookahead on the fast in-process engine to play above the raw policy.

The agent normally plays its policy greedily — one ply, no lookahead. Here, for each legal action it
clones the live battle, reseeds the clone with *fresh* dice (so it never peeks at the real game's
RNG), steps it forward one turn with the opponent modelled by a fixed policy, and scores the
resulting state with the value head — averaged over a few rollouts. It then plays the best-scoring
action. This makes the agent stronger than its own policy *and* than anything it trained against,
for free, by computing instead of recalling — the payoff of having built a 500-battle/sec engine.

It catches tactical blunders the greedy policy makes — e.g. a clause-blocked Sleep Powder rolls
forward to a state that's no better (the foe got a free turn), so it scores below a real attack and
gets dropped — without retraining or a stronger opponent.

Only usable on the engine path (it needs the `ce.Battle` to clone); the browser path would require
reconstructing engine state from poke-env first. This is the measurement/training-judge tool.
"""

from __future__ import annotations

import copy
import math
import random as _random

import torch

from .encoding import encode_global, featurize
from .engine_cpp import Reveal, build_state, reveal_move
from .state import ActionType

import cinnabar_engine as ce  # noqa: E402  (importable only after engine_cpp set the path)


@torch.no_grad()
def _value(net, state, device: str) -> float:
    g = torch.tensor(encode_global(state), dtype=torch.float32, device=device)
    return float(net.value(g))


def _will_fail(action, state) -> bool:
    """True when this move is a DETERMINISTIC no-op right now (Gen 1 mechanics, mirrors the
    encoding's will-fail feature): a primary status move into an already-statused active, or a
    sleep/freeze move while the clause is spent. Search prunes these outright — relying on the
    policy's learned weight proved checkpoint-dependent (browser game 5: Gengar spammed Hypnosis
    x4 into a paralyzed Eggy on models_wf). Pruning provably-null actions is game logic, like not
    searching illegal moves — not a meta prior."""
    if action.type != ActionType.MOVE or not action.effect_status:
        return False
    if action.effect_status == "SLP" and any(m.status == "SLP" for m in state.opponent_team):
        return True
    if action.effect_status == "FRZ" and any(m.status == "FRZ" for m in state.opponent_team):
        return True
    return (action.category == "STATUS" and (action.effect_chance or 0.0) >= 0.999
            and state.opponent_active is not None and bool(state.opponent_active.status))


def _prune_null(candidates: list[int], state) -> list[int]:
    """Drop deterministically-failing moves from the candidate set (keep the set if ALL fail)."""
    if state is None:
        return candidates
    kept = [i for i in candidates
            if i < len(state.available_actions) and not _will_fail(state.available_actions[i], state)]
    return kept or candidates


@torch.no_grad()
def _top_k_candidates(net, state, k: int, device: str) -> list[int]:
    """The k action indices the POLICY rates highest (AlphaZero-style prior gating). Knowledge the
    policy has learned (e.g. 'this sleep will fail under the clause') prunes candidates the value
    head can't tell apart — without it, 1-ply value-leaf search bypasses the policy entirely and
    un-learns its discipline. Also a speedup: fewer candidates = fewer clones."""
    g_feats, a_feats = featurize(state)
    g = torch.tensor(g_feats, dtype=torch.float32, device=device)
    a = torch.tensor(a_feats, dtype=torch.float32, device=device)
    logits = net.score_actions(g, a)
    k = min(k, logits.shape[0])
    return torch.topk(logits, k).indices.tolist()


@torch.no_grad()
def _playout_value(c, player, net, rollout_policy, static, my_spec, opp_spec, device, cap: int) -> float:
    """Play the clone to terminal with `rollout_policy` on both sides; return win(1)/tie(0.5)/loss(0)
    from `player`'s view. The deeper leaf — robust to a miscalibrated value head. 0.5 if it doesn't
    resolve within `cap` turns (rare; gen1 games are ~45 turns)."""
    spec0 = my_spec if player == 0 else opp_spec
    spec1 = opp_spec if player == 0 else my_spec
    r1, r2 = Reveal(), Reveal()
    turns = 0
    while c.result() == ce.Result.Ongoing and turns < cap:
        turns += 1
        s0 = build_state(c, 0, spec0, static, "ro", reveal=r1, opp_team=spec1)
        s1 = build_state(c, 1, spec1, static, "ro_o", reveal=r2, opp_team=spec0)
        a0 = rollout_policy.select_action(s0)
        a1 = rollout_policy.select_action(s1)
        c.step(c.choices(0)[a0.index], c.choices(1)[a1.index])
    res = c.result()
    if res == ce.Result.Tie or res == ce.Result.Ongoing:
        return 0.5
    win = (res == ce.Result.P1Win) if player == 0 else (res == ce.Result.P2Win)
    return 1.0 if win else 0.0


def search_action_values(battle, player, net, opp_model, static, my_spec, opp_spec,
                         reveal=None, device="cpu", rollouts: int = 3, rng=None,
                         leaf: str = "value", rollout_policy=None, rollout_cap: int = 150,
                         state=None, top_k: int = 0) -> tuple[list[int], list[float]]:
    """Per-candidate 1-ply search values for `player`: (candidate action indices, mean leaf values).
    The full Q-vector — expert iteration distils the whole distribution, not just the argmax.
    `search_action_index` is the argmax wrapper. RNG consumption order is identical to the original
    implementation, so prior measurements reproduce exactly."""
    rng = rng or _random
    n = len(battle.choices(player))
    if n <= 1:
        return [0], [0.0]
    candidates = list(range(n))
    if top_k and top_k < n:
        if state is None:
            state = build_state(battle, player, my_spec, static, "srch_pri",
                                reveal=copy.deepcopy(reveal) if reveal is not None else None,
                                opp_team=opp_spec)
        candidates = _top_k_candidates(net, state, top_k, device)
    candidates = _prune_null(candidates, state)

    # Model the opponent's move once (a point estimate; the agent can't see the real opponent policy).
    opp_state = build_state(battle, 1 - player, opp_spec, static, "srch_o", reveal=None, opp_team=my_spec)
    opp_idx = opp_model.select_action(opp_state).index if opp_state.available_actions else 0
    if opp_idx >= len(battle.choices(1 - player)):
        opp_idx = 0

    values: list[float] = []
    for i in candidates:
        total = 0.0
        for _ in range(rollouts):
            c = battle.clone()
            c.reseed(rng.getrandbits(63))  # fresh dice — never replay the live game's RNG
            mine = c.choices(player)[i]
            theirs = c.choices(1 - player)[opp_idx]
            c1, c2 = (mine, theirs) if player == 0 else (theirs, mine)
            c.step(c1, c2)
            if leaf == "rollout" and rollout_policy is not None:
                total += _playout_value(c, player, net, rollout_policy, static, my_spec, opp_spec,
                                        device, rollout_cap)
            else:
                leaf_state = build_state(c, player, my_spec, static, "srch",
                                         reveal=copy.deepcopy(reveal) if reveal is not None else None,
                                         opp_team=opp_spec)  # deepcopy: don't leak future reveals
                total += _value(net, leaf_state, device)
        values.append(total / rollouts)
    return candidates, values


@torch.no_grad()
def _playout_bootstrap(c, player, net, rollout_policy, static, my_spec, opp_spec, device,
                       depth: int) -> float:
    """Play `depth` more turns with `rollout_policy` on BOTH sides, then evaluate the value leaf
    (terminal results return the actual outcome). Buys medium-horizon vision: a one-turn leaf
    can't see an attrition war (browser game 5: three mons fed sequentially to a Recover-stalling
    Starmie that Chansey walls for free) — a 10-turn playout ends inside the war's verdict.
    Use with the CALIBRATED leaf (outcome and leaf share the [0,1] win-prob scale)."""
    spec_p1 = my_spec if player == 0 else opp_spec
    spec_p2 = opp_spec if player == 0 else my_spec
    for _ in range(depth):
        if c.result() != ce.Result.Ongoing:
            break
        s1 = build_state(c, 0, spec_p1, static, "bo", reveal=None, opp_team=spec_p2)
        s2 = build_state(c, 1, spec_p2, static, "bo_o", reveal=None, opp_team=spec_p1)
        a1 = rollout_policy.select_action(s1)
        a2 = rollout_policy.select_action(s2)
        c.step(c.choices(0)[a1.index], c.choices(1)[a2.index])
    res = c.result()
    if res != ce.Result.Ongoing:
        if res == ce.Result.Tie:
            return 0.5
        return 1.0 if (res == ce.Result.P1Win) == (player == 0) else 0.0
    leaf_state = build_state(c, player, my_spec, static, "bo_l", reveal=None, opp_team=opp_spec)
    return _value(net, leaf_state, device)


def search_action_values_minimax(battle, player, net, static, my_spec, opp_spec,
                                 reveal=None, device="cpu", rollouts: int = 2, rng=None,
                                 state=None, top_k: int = 3, opp_top_k: int = 0,
                                 paranoia: float = 1.0, leaf_depth: int = 0,
                                 rollout_policy=None,
                                 opp_temp: float = 0.0) -> tuple[list[int], list[float]]:
    """ADVERSARIAL one-turn lookahead: for each of MY gated candidates, roll the turn forward
    against EVERY plausible opponent reply (their policy top-k from their view; 0 = all their
    legal actions) and score each leaf; my candidate's value = paranoia * worst-case +
    (1 - paranoia) * mean.

    Why: the point-estimate opponent model is a coward — it never Explodes, never sleeps
    strategically, never sacrifices — so the agent walks into every play a human telegraphs
    (browser ground truth, 2026-06-09: fresh Chansey fed to an obvious Explosion; Snorlax switched
    into Sleep Powder; Tauros fed to Eggy). Worst-case over the reply set prices the threat the
    moment it exists: the sleep-absorber switch and the Explosion dodge fall out of minimax, not
    hand-coded rules. Cost: ~|mine| * |theirs| * rollouts clones per decision — fine at browser
    speeds, ~10x the point-estimate search as an engine judge."""
    rng = rng or _random
    n = len(battle.choices(player))
    if n <= 1:
        return [0], [0.0]
    candidates = list(range(n))
    if top_k and top_k < n:
        if state is None:
            state = build_state(battle, player, my_spec, static, "srch_pri",
                                reveal=copy.deepcopy(reveal) if reveal is not None else None,
                                opp_team=opp_spec)
        candidates = _top_k_candidates(net, state, top_k, device)
    candidates = _prune_null(candidates, state)

    opp_n = len(battle.choices(1 - player))
    opp_cands = list(range(opp_n))
    if opp_top_k and opp_top_k < opp_n:
        opp_state = build_state(battle, 1 - player, opp_spec, static, "srch_o",
                                reveal=None, opp_team=my_spec)
        opp_cands = _top_k_candidates(net, opp_state, opp_top_k, device)

    values: list[float] = []
    for i in candidates:
        per_reply: list[float] = []
        for j in opp_cands:
            total = 0.0
            for _ in range(rollouts):
                c = battle.clone()
                c.reseed(rng.getrandbits(63))
                mine = c.choices(player)[i]
                theirs = c.choices(1 - player)[j]
                c1, c2 = (mine, theirs) if player == 0 else (theirs, mine)
                c.step(c1, c2)
                if leaf_depth > 0 and rollout_policy is not None:
                    total += _playout_bootstrap(c, player, net, rollout_policy, static,
                                                my_spec, opp_spec, device, leaf_depth)
                else:
                    leaf_state = build_state(c, player, my_spec, static, "srch",
                                             reveal=copy.deepcopy(reveal) if reveal is not None else None,
                                             opp_team=opp_spec)
                    total += _value(net, leaf_state, device)
            per_reply.append(total / rollouts)
        if opp_temp > 0:
            # QUANTAL-RESPONSE opponent: weight each reply by how good it is FOR THEM
            # (their value = 1 - mine, zero-sum), softmax at temperature opp_temp. The paranoia
            # scalar failed from both ends (browser game 8: p=0.5 averaged away a telegraphed
            # Explosion; p=1.0 hedged into the switch carousel). Threat-weighting prices the
            # Explosion at ~full weight exactly when it IS their best move, and prices phantom
            # threats they'd never click at ~zero — real nukes dodged, no worst-case hedging.
            mx = max(1.0 - v for v in per_reply)
            ws = [math.exp(((1.0 - v) - mx) / opp_temp) for v in per_reply]
            tot_w = sum(ws)
            values.append(sum(w * v for w, v in zip(ws, per_reply)) / tot_w)
        else:
            worst = min(per_reply)
            mean = sum(per_reply) / len(per_reply)
            values.append(paranoia * worst + (1.0 - paranoia) * mean)
    return candidates, values


def search_action_index(battle, player, net, opp_model, static, my_spec, opp_spec,
                        reveal=None, device="cpu", rollouts: int = 3, rng=None,
                        leaf: str = "value", rollout_policy=None, rollout_cap: int = 150,
                        state=None, top_k: int = 0) -> int:
    """Best action index for `player` by 1-ply lookahead (argmax of `search_action_values`).
    `leaf="value"` scores the rolled-forward state with the value head (fast); `leaf="rollout"`
    plays it out to terminal with `rollout_policy` (deeper, slower). Opponent modelled by
    `opp_model`; transitions averaged over `rollouts`.

    `top_k > 0` enables POLICY-PRIOR gating: only the policy's top-k actions for `state` (the
    pre-built BattleState for `player`; built here if not passed) are searched. The policy proposes,
    the value head disposes — policy-learned discipline transfers into search instead of being
    bypassed by it. `top_k=0` keeps the original search-everything behaviour."""
    candidates, values = search_action_values(
        battle, player, net, opp_model, static, my_spec, opp_spec,
        reveal=reveal, device=device, rollouts=rollouts, rng=rng,
        leaf=leaf, rollout_policy=rollout_policy, rollout_cap=rollout_cap,
        state=state, top_k=top_k)
    best = max(range(len(values)), key=values.__getitem__)  # first-max tie-break, as before
    return candidates[best]


def selfplay_search_battle(net, opp_model, team1, team2, static, seed, clauses: bool = False,
                           turn_limit: int = 300, device: str = "cpu", rollouts: int = 3, rng=None,
                           observer=None, top_k: int = 0):
    """Both sides choose moves by search with the same `net`. Returns the ce.Battle. The strong-judge
    battle for team evaluation — each team is piloted as well as value-head lookahead can manage, so
    a team's win-rate reflects its strength under a stronger judge than the raw greedy policy.
    `observer`, if given, is called pre-step each turn as observer(battle, s1, a1) — a diagnostics
    hook on the P1 side (must not mutate)."""
    spec1 = [(s, list(m)) for s, m in team1]
    spec2 = [(s, list(m)) for s, m in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, "ss", reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, "ss_o", reveal=r2, opp_team=spec1)
        i1 = search_action_index(battle, 0, net, opp_model, static, spec1, spec2,
                                 reveal=r1, device=device, rollouts=rollouts, rng=rng,
                                 state=s1, top_k=top_k)
        i2 = search_action_index(battle, 1, net, opp_model, static, spec2, spec1,
                                 reveal=r2, device=device, rollouts=rollouts, rng=rng,
                                 state=s2, top_k=top_k)
        a1, a2 = s1.available_actions[i1], s2.available_actions[i2]
        if observer is not None:
            observer(battle, s1, a1)
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle


def play_search_battle(net, opp_policy, opp_model, team1, team2, static, seed,
                       tag: str = "srch", turn_limit: int = 1000, clauses: bool = False,
                       device: str = "cpu", rollouts: int = 3, rng=None,
                       leaf: str = "value", rollout_policy=None, stats: dict | None = None,
                       observer=None, top_k: int = 0, minimax: bool = False,
                       opp_top_k: int = 0, paranoia: float = 1.0, leaf_depth: int = 0,
                       opp_temp: float = 0.0):
    """p1 plays by decision-time search (value head `net` + `opp_model` for the lookahead); p2 plays
    `opp_policy`. `stats`, if given, accumulates sleep-clause discipline. `observer`, if given, is
    called pre-step each turn as observer(battle, s1, a1) — a diagnostics hook (must not mutate).
    Returns the ce.Battle."""
    spec1 = [(s, list(mvs)) for s, mvs in team1]
    spec2 = [(s, list(mvs)) for s, mvs in team2]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, tag, reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, tag + "_opp", reveal=r2, opp_team=spec1)
        if minimax:
            cands, vals = search_action_values_minimax(battle, 0, net, static, spec1, spec2,
                                                       reveal=r1, device=device, rollouts=rollouts,
                                                       rng=rng, state=s1, top_k=top_k,
                                                       opp_top_k=opp_top_k, paranoia=paranoia,
                                                       leaf_depth=leaf_depth,
                                                       rollout_policy=rollout_policy,
                                                       opp_temp=opp_temp)
            i1 = cands[max(range(len(vals)), key=vals.__getitem__)]
        else:
            i1 = search_action_index(battle, 0, net, opp_model, static, spec1, spec2,
                                     reveal=r1, device=device, rollouts=rollouts, rng=rng,
                                     leaf=leaf, rollout_policy=rollout_policy,
                                     state=s1, top_k=top_k)
        a1 = s1.available_actions[i1]
        if stats is not None and any(getattr(x, "effect_status", "") == "SLP" for x in s1.available_actions) \
                and any(m.status == "SLP" for m in s1.opponent_team):
            stats["tempt"] = stats.get("tempt", 0) + 1
            if getattr(a1, "effect_status", "") == "SLP":
                stats["reslept"] = stats.get("reslept", 0) + 1
        a2 = opp_policy.select_action(s2)
        if observer is not None:
            observer(battle, s1, a1)
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle
