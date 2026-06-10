"""Parity tests for the C++ fast path inside cinnabar.search.

search_action_values / search_action_values_minimax take a fast branch (one
ce.search_leaves call + one batched value forward) when the opponent-model /
rollout pilots have C++ twins. These tests pin that branch against the original
per-leaf Python path on identical positions with identical dice: same candidate
sets, values equal to float tolerance (batched vs single forwards differ only in
summation order), and exactly equal where no net is involved (playout leaves).
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("cinnabar.engine_cpp", reason="C++ engine module not built")
ce = pytest.importorskip("cinnabar_engine")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import cinnabar.search as S  # noqa: E402
from cinnabar.encoding import ACTION_DIM, GLOBAL_DIM  # noqa: E402
from cinnabar.engine_cpp import Reveal, StaticData, build_state, register_encoder  # noqa: E402
from cinnabar.policy import SmartHeuristicPolicy  # noqa: E402
from cinnabar.rl.net import ActionScorer  # noqa: E402

TEAM_A = [("Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]),
          ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
          ("Exeggutor", ["Psychic", "Sleep Powder", "Explosion", "Body Slam"]),
          ("Starmie", ["Thunderbolt", "Ice Beam", "Recover", "Thunder Wave"])]
TEAM_B = [("Zapdos", ["Thunderbolt", "Drill Peck", "Thunder Wave", "Rest"]),
          ("Gengar", ["Psychic", "Thunderbolt", "Explosion", "Thunder Wave"]),
          ("Chansey", ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
          ("Jynx", ["Lovely Kiss", "Blizzard", "Psychic", "Rest"])]


@pytest.fixture(scope="module")
def env():
    static = StaticData(1)
    register_encoder(static)
    torch.manual_seed(7)
    net = ActionScorer(GLOBAL_DIM, ACTION_DIM, 64)
    net.eval()
    return static, net


def _positions(n_turns=10, seed=11):
    """A battle advanced by random play, with the P1 Reveal memory maintained."""
    spec1 = [(s, list(m)) for s, m in TEAM_A]
    spec2 = [(s, list(m)) for s, m in TEAM_B]
    b = ce.make_battle(spec1, spec2, seed)
    b.set_clauses(True)
    r1 = Reveal()
    rng = random.Random(seed)
    static = StaticData(1)
    for _ in range(n_turns):
        if b.result() != ce.Result.Ongoing:
            break
        s1 = build_state(b, 0, spec1, static, "t", reveal=r1, opp_team=spec2)
        s2 = build_state(b, 1, spec2, static, "t", reveal=None, opp_team=spec1)
        i1 = rng.randrange(len(s1.available_actions))
        i2 = rng.randrange(len(s2.available_actions))
        from cinnabar.engine_cpp import reveal_move
        reveal_move(r1, s2, s2.available_actions[i2])
        ce.step_pair(b, i1, i2, None, None)
    return b, spec1, spec2, r1


def _both(env, fn):
    """Run fn() on the fast path and with the fast branch disabled; return both results."""
    fast = fn()
    orig = S._heuristic_kind
    S._heuristic_kind = lambda p: None
    try:
        slow = fn()
    finally:
        S._heuristic_kind = orig
    return fast, slow


@pytest.mark.parametrize("leaf", ["value", "rollout"])
def test_point_estimate_parity(env, leaf):
    static, net = env
    b, spec1, spec2, r1 = _positions()
    if b.result() != ce.Result.Ongoing:
        pytest.skip("battle ended early")
    opp_model = SmartHeuristicPolicy()
    rp = SmartHeuristicPolicy() if leaf == "rollout" else None

    def go():
        return S.search_action_values(
            b, 0, net, opp_model, static, spec1, spec2, reveal=r1, rollouts=3,
            rng=random.Random(99), leaf=leaf, rollout_policy=rp, top_k=3)

    (fc, fv), (sc, sv) = _both(env, go)
    assert fc == sc
    if leaf == "rollout":  # no net in the leaves: must be exactly equal
        assert fv == sv
    else:
        np.testing.assert_allclose(fv, sv, atol=1e-5)


@pytest.mark.parametrize("config", [dict(opp_temp=0.0, paranoia=1.0),
                                    dict(opp_temp=0.35),
                                    dict(opp_temp=0.35, leaf_depth=8)])
def test_minimax_parity(env, config):
    static, net = env
    b, spec1, spec2, r1 = _positions(seed=13)
    if b.result() != ce.Result.Ongoing:
        pytest.skip("battle ended early")
    rp = SmartHeuristicPolicy() if config.get("leaf_depth") else None

    def go():
        return S.search_action_values_minimax(
            b, 0, net, static, spec1, spec2, reveal=r1, rollouts=2,
            rng=random.Random(55), top_k=3, opp_top_k=3, rollout_policy=rp, **config)

    (fc, fv), (sc, sv) = _both(env, go)
    assert fc == sc
    np.testing.assert_allclose(fv, sv, atol=1e-5)
