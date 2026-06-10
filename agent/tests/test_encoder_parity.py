"""Exact-parity tests for the C++ observation encoder and heuristic pilots.

The C++ encoder (engine/src/encoder.cpp) is a throughput rewrite of build_state +
featurize; these tests assert it produces BIT-IDENTICAL float32 features and
decision-identical heuristic picks across whole battles — partial info (Reveal vs
Observer), clauses, generated movesets, duplicate-move edge cases. Any divergence
is a bug in the transcription, never tolerance-worthy.

Needs the built engine module; skipped otherwise (the rest of the suite is engine-free).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("cinnabar.engine_cpp", reason="C++ engine module not built")
ce = pytest.importorskip("cinnabar_engine")  # importable once engine_cpp set sys.path
np = pytest.importorskip("numpy")

from cinnabar.encoding import featurize  # noqa: E402
from cinnabar.engine_cpp import (  # noqa: E402
    Reveal, StaticData, build_state, load_teams, register_encoder, reveal_move)
from cinnabar.policy import MaxDamagePolicy, SmartHeuristicPolicy, StallerPolicy  # noqa: E402

TEAMS_DIR = Path(__file__).resolve().parents[2] / "teams"

FALLBACK_1 = [("Tauros", ["Body Slam", "Earthquake", "Blizzard", "Hyper Beam"]),
              ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
              ("Exeggutor", ["Psychic", "Sleep Powder", "Explosion", "Body Slam"]),
              ("Starmie", ["Thunderbolt", "Ice Beam", "Recover", "Thunder Wave"]),
              ("Chansey", ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"]),
              ("Alakazam", ["Psychic", "Thunder Wave", "Recover", "Seismic Toss"])]
FALLBACK_2 = [("Zapdos", ["Thunderbolt", "Drill Peck", "Thunder Wave", "Rest"]),
              ("Rhydon", ["Earthquake", "Body Slam", "Blizzard", "Fire Blast"]),
              ("Lapras", ["Blizzard", "Surf", "Body Slam", "Rest"]),
              ("Gengar", ["Psychic", "Thunderbolt", "Explosion", "Thunder Wave"]),
              ("Snorlax", ["Body Slam", "Earthquake", "Hyper Beam", "Rest"]),
              ("Jynx", ["Blizzard", "Psychic", "Body Slam", "Seismic Toss"])]
# Duplicate move in one moveset: Python's revealed-move profile expands the seen-ID set
# over the whole moveset (both copies count once either is seen) — the C++ encoder
# reproduces that quirk and this team would catch a slot-mask-only implementation.
DUP_TEAM = [("Starmie", ["Surf", "Surf", "Blizzard", "Recover"]),
            ("Snorlax", ["Body Slam", "Hyper Beam", "Rest", "Earthquake"]),
            ("Jynx", ["Lovely Kiss", "Blizzard", "Psychic", "Rest"])]

KINDS = {"maxdamage": MaxDamagePolicy(), "smart": SmartHeuristicPolicy(),
         "staller": StallerPolicy()}


@pytest.fixture(scope="module")
def static() -> StaticData:
    s = StaticData(1)
    register_encoder(s)
    return s


def _compare_battle(team1, team2, seed, static, *, clauses=False, turn_cap=400,
                    heuristics=True) -> int:
    """Random-play a battle; every turn assert C++ encode == Python featurize(build_state)
    for both sides (exact float32 equality) and that the C++ pilots pick the same actions
    as their Python twins. Returns states compared."""
    spec1 = [(s, list(m)) for s, m in team1]
    spec2 = [(s, list(m)) for s, m in team2]
    b = ce.make_battle(spec1, spec2, seed)
    if clauses:
        b.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    o1, o2 = ce.Observer(), ce.Observer()
    rng = random.Random(seed * 31 + 7)
    checked = 0
    turns = 0
    while b.result() == ce.Result.Ongoing and turns < turn_cap:
        turns += 1
        picks = []
        for p, mine, theirs, rev, obs in ((0, spec1, spec2, r1, o1), (1, spec2, spec1, r2, o2)):
            s_py = build_state(b, p, mine, static, "t", reveal=rev, opp_team=theirs)
            g_py, a_py = featurize(s_py)
            g_c, a_c = ce.encode(b, p, obs)
            np.testing.assert_array_equal(np.asarray(g_py, np.float32), g_c,
                                          err_msg=f"global mismatch p{p} turn {turns} seed {seed}")
            np.testing.assert_array_equal(np.asarray(a_py, np.float32), a_c,
                                          err_msg=f"action mismatch p{p} turn {turns} seed {seed}")
            if heuristics:
                for kind, pol in KINDS.items():
                    assert pol.select_action(s_py).index == ce.select_heuristic(b, p, obs, kind), \
                        f"{kind} pick mismatch p{p} turn {turns} seed {seed}"
            picks.append((s_py, rng.randrange(len(s_py.available_actions))))
            checked += 1
        (s1, i1), (s2, i2) = picks
        reveal_move(r1, s2, s2.available_actions[i2])
        reveal_move(r2, s1, s1.available_actions[i1])
        ce.step_pair(b, i1, i2, o1, o2)  # steps the engine AND mirrors reveal_move into o1/o2
    return checked


def test_parity_fallback_teams(static):
    total = sum(_compare_battle(FALLBACK_1, FALLBACK_2, seed, static) for seed in (1, 2, 3))
    assert total > 100  # sanity: battles actually ran


def test_parity_with_clauses(static):
    total = sum(_compare_battle(FALLBACK_1, FALLBACK_1, seed, static, clauses=True)
                for seed in (11, 12))
    assert total > 50


def test_parity_teams_pool(static):
    teams = load_teams(TEAMS_DIR)
    if not teams:
        pytest.skip("teams/ pool not present")
    rng = random.Random(0)
    total = 0
    for seed in range(4):
        t1, t2 = rng.choice(teams), rng.choice(teams)
        total += _compare_battle(t1, t2, 100 + seed, static, clauses=True)
    assert total > 100


def test_parity_generated_movesets(static):
    """Whole-metagame generated teams: stresses move metadata far beyond the curated pool."""
    from cinnabar import movesets

    rng = random.Random(42)
    for seed in range(3):
        t1, t2 = movesets.generate_team(rng), movesets.generate_team(rng)
        _compare_battle(t1, t2, 200 + seed, static, clauses=seed % 2 == 0)


def test_parity_duplicate_moves(static):
    _compare_battle(DUP_TEAM, FALLBACK_1[:3], 7, static)


def test_encode_batch_matches_encode(static):
    """The batched entry point must equal per-battle encode + the Python material sums."""
    spec1 = [(s, list(m)) for s, m in FALLBACK_1]
    spec2 = [(s, list(m)) for s, m in FALLBACK_2]
    battles, observers = [], []
    rng = random.Random(5)
    for i in range(8):
        b = ce.make_battle(spec1, spec2, 500 + i)
        o1, o2 = ce.Observer(), ce.Observer()
        for _ in range(rng.randrange(12)):  # desync the battles
            if b.result() != ce.Result.Ongoing:
                break
            n1, n2 = len(b.choices(0)), len(b.choices(1))
            ce.step_pair(b, rng.randrange(n1), rng.randrange(n2), o1, o2)
        if b.result() == ce.Result.Ongoing:
            battles.append(b)
            observers.append((o1, o2))

    for player in (0, 1):
        obs = [o[player] for o in observers]
        # Clone observers so the batched call sees the same pre-reveal state as encode().
        obs_single = [o.clone() for o in obs]
        glob, act, mask, my_mat, opp_mat = ce.encode_batch(
            battles, [player] * len(battles), obs)
        for i, b in enumerate(battles):
            g, a = ce.encode(b, player, obs_single[i])
            np.testing.assert_array_equal(glob[i], g)
            n = a.shape[0]
            assert mask[i, :n].all() and not mask[i, n:].any()
            np.testing.assert_array_equal(act[i, :n], a)
            assert not act[i, n:].any()
            # Material parity with the Python adapter's bookkeeping.
            spec = spec1 if player == 0 else spec2
            other = spec2 if player == 0 else spec1
            rev = Reveal()
            rev.mons = {j for j in range(6) if obs_single[i].mons >> j & 1}
            s_py = build_state(b, player, spec, static, "t", reveal=rev, opp_team=other)
            assert my_mat[i] == pytest.approx(sum(m.hp_fraction for m in s_py.team), abs=0)
            assert opp_mat[i] == pytest.approx(
                sum(m.hp_fraction for m in s_py.opponent_team), abs=0)
