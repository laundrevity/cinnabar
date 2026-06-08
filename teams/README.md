# Teams

A **pool** of Gen 1 OU teams in [Pokémon Showdown export format](https://pokepast.es/syntax)
(the text from the teambuilder's Import/Export button). Gen 1 has no abilities, items,
natures, or EV/IV spreads to specify, so each set is just a species (optionally with gender)
and up to four moves.

`agent/cinnabar/teams.py` loads **every `*.txt` in this directory** and a random one is used
per battle (for both the agent and its opponents) during training, evaluation, and play. That
team variety is what stops the agent from being exploitable by memorising a single roster.

Current pool (16 teams, deliberately archetype-diverse so the agent must generalize rather than
memorize a roster — most keep a Gen 1 OU staple core of Tauros / Snorlax / Chansey / Starmie but
vary the lead, win condition, and 5th/6th slots):

Standard / offense:
- `gen1ou-sample.txt` — Alakazam + Exeggutor (the default standard team)
- `gen1ou-rhydon.txt` — Rhydon ground core, Reflect Snorlax
- `gen1ou-zapdos.txt` — Zapdos + Rhydon
- `gen1ou-jynx.txt` — Jynx sleep lead + Lapras
- `gen1ou-gengar.txt` — Alakazam + Gengar psychic/ghost
- `gen1ou-persian.txt` — Persian (Slash / Bubble Beam) lead

Trappers (partial-trap stall-breakers):
- `gen1ou-cloyster.txt` — Cloyster Clamp + Explosion
- `gen1ou-dragonite.txt` — Dragonite Wrap + Agility
- `gen1ou-moltres.txt` — Moltres Fire Blast / Fire Spin
- `gen1ou-victreebel.txt` — Victreebel Sleep Powder / Wrap, double sleep with Jynx

Setup / sweepers:
- `gen1ou-slowbro.txt` — Slowbro Amnesia stall
- `gen1ou-articuno.txt` — Articuno + Zapdos Agility birds
- `gen1ou-jolteon.txt` — Jolteon (Pin Missile / Double Kick) paralysis
- `gen1ou-hypno.txt` — Hypno hypnosis + Golem

Status / control:
- `gen1ou-stall.txt` — Toxic Chansey + Leech Seed Exeggutor defensive core
- `gen1ou-gengar-disable.txt` — Disable Gengar + Counter Chansey tech

These exercise the full validated move set (partial-trap, multi-hit, drain, Amnesia, Toxic,
Leech Seed, Disable, screens, the high-crit and secondary-effect moves), all bit-for-bit vs
Showdown — validate any new move with `engine/tools/validate_moves.py` before adding it.

Add a team by dropping another `*.txt` here. **Team-building** (the agent choosing its own
team) remains a separate, later problem — this is team *variety*, not team *construction*.
