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
import random as _random

import torch

from .encoding import encode_global
from .engine_cpp import Reveal, build_state, reveal_move

import cinnabar_engine as ce  # noqa: E402  (importable only after engine_cpp set the path)


@torch.no_grad()
def _value(net, state, device: str) -> float:
    g = torch.tensor(encode_global(state), dtype=torch.float32, device=device)
    return float(net.value(g))


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


def search_action_index(battle, player, net, opp_model, static, my_spec, opp_spec,
                        reveal=None, device="cpu", rollouts: int = 3, rng=None,
                        leaf: str = "value", rollout_policy=None, rollout_cap: int = 150) -> int:
    """Best action index for `player` by 1-ply lookahead. `leaf="value"` scores the rolled-forward
    state with the value head (fast); `leaf="rollout"` plays it out to terminal with `rollout_policy`
    (deeper, slower). Opponent modelled by `opp_model`; transitions averaged over `rollouts`."""
    rng = rng or _random
    n = len(battle.choices(player))
    if n <= 1:
        return 0

    # Model the opponent's move once (a point estimate; the agent can't see the real opponent policy).
    opp_state = build_state(battle, 1 - player, opp_spec, static, "srch_o", reveal=None, opp_team=my_spec)
    opp_idx = opp_model.select_action(opp_state).index if opp_state.available_actions else 0
    if opp_idx >= len(battle.choices(1 - player)):
        opp_idx = 0

    best_i, best_v = 0, float("-inf")
    for i in range(n):
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
        v = total / rollouts
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def selfplay_search_battle(net, opp_model, team1, team2, static, seed, clauses: bool = False,
                           turn_limit: int = 300, device: str = "cpu", rollouts: int = 3, rng=None):
    """Both sides choose moves by search with the same `net`. Returns the ce.Battle. The strong-judge
    battle for team evaluation — each team is piloted as well as value-head lookahead can manage, so
    a team's win-rate reflects its strength under a stronger judge than the raw greedy policy."""
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
                                 reveal=r1, device=device, rollouts=rollouts, rng=rng)
        i2 = search_action_index(battle, 1, net, opp_model, static, spec2, spec1,
                                 reveal=r2, device=device, rollouts=rollouts, rng=rng)
        a1, a2 = s1.available_actions[i1], s2.available_actions[i2]
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle


def play_search_battle(net, opp_policy, opp_model, team1, team2, static, seed,
                       tag: str = "srch", turn_limit: int = 1000, clauses: bool = False,
                       device: str = "cpu", rollouts: int = 3, rng=None,
                       leaf: str = "value", rollout_policy=None, stats: dict | None = None):
    """p1 plays by decision-time search (value head `net` + `opp_model` for the lookahead); p2 plays
    `opp_policy`. `stats`, if given, accumulates sleep-clause discipline. Returns the ce.Battle."""
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
        i1 = search_action_index(battle, 0, net, opp_model, static, spec1, spec2,
                                 reveal=r1, device=device, rollouts=rollouts, rng=rng,
                                 leaf=leaf, rollout_policy=rollout_policy)
        a1 = s1.available_actions[i1]
        if stats is not None and any(getattr(x, "effect_status", "") == "SLP" for x in s1.available_actions) \
                and any(m.status == "SLP" for m in s1.opponent_team):
            stats["tempt"] = stats.get("tempt", 0) + 1
            if getattr(a1, "effect_status", "") == "SLP":
                stats["reslept"] = stats.get("reslept", 0) + 1
        a2 = opp_policy.select_action(s2)
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle
