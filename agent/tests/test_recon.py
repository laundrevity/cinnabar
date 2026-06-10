"""Round-trip tests for the browser-ground-truth reconstruction (no Showdown server needed).

Stub poke-env-shaped objects -> recon.reconstruct -> assert the engine battle mirrors the
position. Requires the built engine module (skipped otherwise) + poke-env's data tables
(StaticData), both present in the agent uv env.
"""

from types import SimpleNamespace

import pytest

ce = pytest.importorskip("cinnabar_engine", reason="engine not built (engine/build)")

from cinnabar.engine_cpp import StaticData  # noqa: E402
from cinnabar.recon import PLACEHOLDER, ReconError, reconstruct  # noqa: E402


class _Status:
    def __init__(self, name):
        self.name = name


def _mon(species, moves, hp=1.0, status=None, fainted=False, active=False,
         boosts=None, status_counter=0):
    return SimpleNamespace(
        species=species,
        moves={m: None for m in moves},
        current_hp_fraction=hp,
        status=_Status(status) if status else None,
        fainted=fainted,
        active=active,
        boosts=boosts or {},
        effects={},
        status_counter=status_counter,
    )


def _battle(team, opp_team):
    return SimpleNamespace(
        team={f"p1: {i}": m for i, m in enumerate(team)},
        opponent_team={f"p2: {i}": m for i, m in enumerate(opp_team)},
        active_pokemon=next((m for m in team if m.active), None),
        turn=7,
    )


@pytest.fixture(scope="module")
def static():
    return StaticData(1)


def test_roundtrip_position(static):
    ours = [
        _mon("tauros", ["bodyslam", "hyperbeam", "earthquake", "blizzard"], hp=0.5,
             status="PAR", active=True),
        _mon("snorlax", ["bodyslam", "selfdestruct", "rest", "icebeam"], hp=1.0),
    ]
    theirs = [
        _mon("chansey", ["icebeam"], hp=0.89, status="FRZ", active=True),
    ]
    battle, spec1, spec2 = reconstruct(_battle(ours, theirs), static)

    ts1 = battle.team_state(0)
    assert ts1[0][0] == "Tauros" and ts1[0][4] is True       # our active
    assert abs(ts1[0][1] - 0.5) < 0.01                       # injected HP
    assert ts1[0][2] == "PAR"  # team_state reports the adapter's uppercase status codes
    assert ts1[1][0] == "Snorlax" and ts1[1][1] == 1.0

    ts2 = battle.team_state(1)
    assert ts2[0][0] == "Chansey" and ts2[0][2] == "FRZ" and ts2[0][4] is True
    assert abs(ts2[0][1] - 0.89) < 0.01
    assert len(ts2) == 6                                     # padded to a full team
    assert ts2[1][0] == PLACEHOLDER[0]                       # unrevealed slots = placeholder

    # The revealed move survives; the rest of Chansey's set is movepool padding.
    assert any(m.lower().replace(" ", "").replace("-", "") == "icebeam" for m in spec2[0][1])
    assert len(spec2[0][1]) == 4

    # The position is playable: we have legal choices and stepping works.
    assert len(battle.choices(0)) > 0
    assert battle.result() == ce.Result.Ongoing


def test_boosts_and_faints(static):
    ours = [
        _mon("alakazam", ["psychic", "recover", "thunderwave", "seismictoss"], hp=0.8,
             active=True, boosts={"spa": 2}),
        _mon("starmie", ["surf", "recover", "thunderwave", "blizzard"], hp=0.0, fainted=True),
    ]
    theirs = [_mon("snorlax", ["bodyslam"], hp=0.55, active=True)]
    battle, spec1, _ = reconstruct(_battle(ours, theirs), static)
    ts = battle.team_state(0)
    assert ts[1][3] is True            # Starmie fainted
    assert battle.active_boosts(0)[2] == 2  # spc stage injected (atk, def, spc, spe)


def test_unmappable_raises(static):
    with pytest.raises(ReconError):
        reconstruct(_battle([], []), static)
