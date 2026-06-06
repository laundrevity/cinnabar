# Teams

A **pool** of Gen 1 OU teams in [Pokémon Showdown export format](https://pokepast.es/syntax)
(the text from the teambuilder's Import/Export button). Gen 1 has no abilities, items,
natures, or EV/IV spreads to specify, so each set is just a species (optionally with gender)
and up to four moves.

`agent/cinnabar/teams.py` loads **every `*.txt` in this directory** and a random one is used
per battle (for both the agent and its opponents) during training, evaluation, and play. That
team variety is what stops the agent from being exploitable by memorising a single roster.

Current pool (all built around the Gen 1 OU staples — Tauros / Snorlax / Chansey / Starmie —
with different supporting cores):

- `gen1ou-sample.txt` — Alakazam + Exeggutor (the default standard team)
- `gen1ou-rhydon.txt` — Rhydon ground core, Reflect Snorlax
- `gen1ou-zapdos.txt` — Zapdos + Rhydon
- `gen1ou-jynx.txt` — Jynx sleep lead + Lapras
- `gen1ou-gengar.txt` — Alakazam + Gengar psychic/ghost

Add a team by dropping another `*.txt` here. **Team-building** (the agent choosing its own
team) remains a separate, later problem — this is team *variety*, not team *construction*.
