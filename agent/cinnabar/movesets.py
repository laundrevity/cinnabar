"""Per-battle randomized movesets.

The fixed `teams/` pool pins each species to one moveset, so species identity fully
determines the moves — which makes the agent's revealed-team/revealed-moves memory
largely idle (seeing "Starmie" already tells you its four moves). This module keeps the
*rosters* (which six species form a team — realistic compositions stay intact) but
re-samples each species' four moves per battle from a curated movepool, so the same
species runs different sets. That turns the opponent's moveset into real hidden
information the agent must observe and reason about, and forces generalization instead
of memorization.

Every move below is one the C++ engine models bit-for-bit vs Showdown (no flinch /
two-turn / Thrash / Haze / Mimic — those aren't modelled), so any sampled set simulates
correctly with no extra validation. Train with `train_engine.py --random-movesets`.
"""

from __future__ import annotations

import random as _random

# Status (non-damaging) moves — used only to keep a sampled set sane (>=2 attacks per mon).
STATUS_MOVES = {
    "Thunder Wave", "Soft-Boiled", "Recover", "Rest", "Reflect", "Light Screen",
    "Substitute", "Amnesia", "Agility", "Swords Dance", "Sleep Powder", "Stun Spore",
    "Sing", "Lovely Kiss", "Hypnosis", "Confuse Ray", "Toxic", "Disable", "Leech Seed",
}

# Curated Gen 1 OU movepools (realistic options; engine-modelled moves only).
SPECIES_MOVEPOOLS: dict[str, list[str]] = {
    "Tauros":     ["Body Slam", "Hyper Beam", "Earthquake", "Blizzard", "Fire Blast", "Thunderbolt", "Take Down"],
    "Snorlax":    ["Body Slam", "Earthquake", "Hyper Beam", "Self-Destruct", "Ice Beam", "Blizzard",
                   "Surf", "Reflect", "Rest", "Counter", "Amnesia", "Psychic"],
    "Chansey":    ["Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled", "Seismic Toss",
                   "Reflect", "Counter", "Sing", "Toxic", "Psychic"],
    "Starmie":    ["Surf", "Blizzard", "Ice Beam", "Thunderbolt", "Psychic", "Recover", "Thunder Wave"],
    "Alakazam":   ["Psychic", "Seismic Toss", "Thunder Wave", "Recover", "Reflect", "Counter"],
    "Exeggutor":  ["Psychic", "Sleep Powder", "Stun Spore", "Mega Drain", "Explosion",
                   "Double-Edge", "Hyper Beam", "Leech Seed"],
    "Rhydon":     ["Earthquake", "Rock Slide", "Body Slam", "Substitute", "Take Down", "Hyper Beam"],
    "Golem":      ["Earthquake", "Rock Slide", "Body Slam", "Explosion", "Take Down", "Substitute"],
    "Zapdos":     ["Thunderbolt", "Drill Peck", "Thunder Wave", "Agility", "Light Screen", "Reflect"],
    "Jynx":       ["Lovely Kiss", "Blizzard", "Psychic", "Rest", "Ice Beam", "Counter"],
    "Lapras":     ["Blizzard", "Ice Beam", "Thunderbolt", "Body Slam", "Confuse Ray", "Sing", "Surf", "Rest"],
    "Gengar":     ["Hypnosis", "Night Shade", "Thunderbolt", "Mega Drain", "Explosion", "Psychic",
                   "Confuse Ray", "Disable", "Seismic Toss"],
    "Slowbro":    ["Amnesia", "Surf", "Psychic", "Thunder Wave", "Rest", "Ice Beam", "Reflect"],
    "Cloyster":   ["Clamp", "Blizzard", "Hyper Beam", "Explosion", "Ice Beam", "Surf", "Substitute"],
    "Dragonite":  ["Wrap", "Blizzard", "Thunderbolt", "Agility", "Hyper Beam", "Surf", "Body Slam"],
    "Victreebel": ["Sleep Powder", "Stun Spore", "Razor Leaf", "Wrap", "Hyper Beam", "Body Slam",
                   "Mega Drain", "Swords Dance"],
    "Jolteon":    ["Thunderbolt", "Thunder Wave", "Pin Missile", "Double Kick", "Body Slam", "Agility"],
    "Hypno":      ["Hypnosis", "Psychic", "Thunder Wave", "Rest", "Seismic Toss", "Reflect", "Counter"],
    "Articuno":   ["Blizzard", "Ice Beam", "Agility", "Rest", "Reflect", "Hyper Beam"],
    "Moltres":    ["Fire Blast", "Hyper Beam", "Agility", "Fire Spin", "Reflect"],
    "Persian":    ["Slash", "Hyper Beam", "Bubble Beam", "Thunderbolt", "Body Slam", "Substitute"],
}


def sample_moveset(species: str, rng: _random.Random = _random, k: int = 4) -> list[str]:
    """Sample up to `k` distinct moves for a species, guaranteeing >= 2 attacking moves."""
    pool = SPECIES_MOVEPOOLS[species]
    if len(pool) <= k:
        return list(pool)
    pick = rng.sample(pool, k)
    for _ in range(8):  # re-roll a degenerate all-status set (rare; pools are attack-heavy)
        if sum(1 for m in pick if m not in STATUS_MOVES) >= 2:
            break
        pick = rng.sample(pool, k)
    return pick


def sample_team(roster: list[str], rng: _random.Random = _random) -> list[tuple[str, list[str]]]:
    """An engine TeamSpec: the given species, each with a freshly sampled moveset."""
    return [(sp, sample_moveset(sp, rng)) for sp in roster]


def rosters_from_teams(teams) -> list[list[str]]:
    """Extract the species rosters (composition only) from loaded engine TeamSpecs."""
    return [[sp for sp, _ in t] for t in teams]


# Scale-up: generate a whole team from the metagame instead of a fixed roster. The "big four"
# (Tauros / Snorlax / Chansey / Starmie) anchor almost every real RBY OU team, so weight them up
# for realism while still drawing the rest from the full viable pool for maximum variety.
ALL_SPECIES = list(SPECIES_MOVEPOOLS.keys())
BIG_FOUR = ["Tauros", "Snorlax", "Chansey", "Starmie"]


def generate_team(rng: _random.Random = _random, size: int = 6) -> list[tuple[str, list[str]]]:
    """A fresh engine TeamSpec: `size` distinct species sampled from the whole pool (big-four
    weighted), each with a randomly sampled moveset. The scale-up environment."""
    pool = list(ALL_SPECIES)
    chosen: list[str] = []
    for _ in range(min(size, len(pool))):
        weights = [3.0 if sp in BIG_FOUR else 1.0 for sp in pool]
        pick = rng.choices(pool, weights=weights, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return sample_team(chosen, rng)
