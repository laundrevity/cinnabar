"""Team pool loading + a per-battle random teambuilder.

``load_team_strings`` is dependency-free (it just reads files), so it can be
imported and tested without poke-env. ``build_random_teambuilder`` lazily imports
poke-env's ``Teambuilder`` and returns one that yields a random team from the pool
before each battle — the mechanism that gives the agent team variety.
"""

from __future__ import annotations

from pathlib import Path


def load_team_strings(teams_dir: str | Path) -> list[str]:
    """Every ``*.txt`` in `teams_dir`, as Showdown-export team strings."""
    teams = []
    for path in sorted(Path(teams_dir).glob("*.txt")):
        text = path.read_text().strip()
        if text:
            teams.append(text)
    if not teams:
        raise FileNotFoundError(f"no team .txt files found in {teams_dir}")
    return teams


def build_random_teambuilder(team_strings: list[str]):
    """A poke-env Teambuilder that returns a random team from the pool per battle."""
    import random

    from poke_env.teambuilder import Teambuilder

    class RandomTeamFromPool(Teambuilder):
        def __init__(self, teams: list[str]) -> None:
            self.packed = [self.join_team(self.parse_showdown_team(t)) for t in teams]

        def yield_team(self) -> str:
            return random.choice(self.packed)

    return RandomTeamFromPool(team_strings)
