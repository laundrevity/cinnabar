"""Tests for the team-pool loader (dependency-free; no poke-env)."""

from pathlib import Path

import pytest

from cinnabar.teams import load_team_strings

TEAMS_DIR = Path(__file__).resolve().parents[2] / "teams"


def test_loads_multiple_teams():
    teams = load_team_strings(TEAMS_DIR)
    assert len(teams) >= 2


def test_each_team_has_six_pokemon():
    for team in load_team_strings(TEAMS_DIR):
        blocks = [b for b in team.strip().split("\n\n") if b.strip()]
        assert len(blocks) == 6, f"expected 6 Pokémon, got {len(blocks)}"


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_team_strings(tmp_path)
