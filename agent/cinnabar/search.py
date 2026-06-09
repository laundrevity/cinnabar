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


def search_action_index(battle, player, net, opp_model, static, my_spec, opp_spec,
                        reveal=None, device="cpu", rollouts: int = 3, rng=None) -> int:
    """Best action index for `player` by 1-ply lookahead: value-head leaf, opponent modelled by
    `opp_model`, transitions sampled with fresh dice and averaged over `rollouts`."""
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
            leaf = build_state(c, player, my_spec, static, "srch",
                               reveal=copy.deepcopy(reveal) if reveal is not None else None,
                               opp_team=opp_spec)  # deepcopy: don't reveal mons the real memory shouldn't know
            total += _value(net, leaf, device)
        v = total / rollouts
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def play_search_battle(net, opp_policy, opp_model, team1, team2, static, seed,
                       tag: str = "srch", turn_limit: int = 1000, clauses: bool = False,
                       device: str = "cpu", rollouts: int = 3, rng=None):
    """p1 plays by decision-time search (value head `net` + `opp_model` for the lookahead); p2 plays
    `opp_policy`. Returns the ce.Battle so the caller can read result()."""
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
                                 reveal=r1, device=device, rollouts=rollouts, rng=rng)
        a1 = s1.available_actions[i1]
        a2 = opp_policy.select_action(s2)
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])
    return battle
