"""The Exeggutor probe — is the agent's Eggy aversion a piloting-depth gap?

Background (HANDOFF.md): search-judged team evolution restored Snorlax/Chansey/a-sleeper, but the
agent still picks Jynx/Gengar over Exeggutor as the sleeper. Hypothesis (UN-measured until this
probe): Eggy's value is *positional* — bulk, repeated switch-ins, sleep-target selection, Explosion
*timing* — multi-turn value that neither the greedy policy nor 1-ply search can see, whereas Jynx
delivers immediate "click the nuke" value the agent does capture.

The probe: an A/B where the ONLY difference is the sleeper slot. Team A = probe_teams/strong.txt
(Eggy: Sleep Powder / Psychic / Explosion / Double-Edge). Team B = the same team with Jynx
(Lovely Kiss / Blizzard / Psychic / Rest) in Eggy's slot. Each pilot (raw policy, search) plays both
variants over IDENTICAL matchups (same opponent teams, same engine seeds), so win-rate deltas are
the sleeper + how it's piloted, not luck. Alongside win-rate it logs the sleeper's *behaviour*:
turns on field, sleep clicks (and clause-wasted ones), whether the sleep ever lands and on whom,
Explosion clicks + the HP context they happen at, faints, KOs while active, and a damage proxy.

How to read it:
  - Jynx-team >> Eggy-team for the same pilot, AND Eggy never/badly Explodes or wastes its sleep
    → piloting-depth hypothesis HOLDS (the agent can't extract Eggy's positional value).
  - Eggy-team ≈ Jynx-team → the aversion is a JUDGE artifact, not a piloting gap — look at the
    team-evolution fitness signal instead.
  - Search-pilot closes the Eggy gap that the raw pilot shows → 1-ply lookahead already captures
    some of the positional value; deeper search is the lever.

These teams are diagnostic instruments, not training priors (same rule as diagnose_policy.py).

    cd agent
    uv run python probe_eggy.py --ckpt models_clauses/pg_best.pt --battles 200 --clauses
    # search pilot is slow (search every move); --pilots raw for a quick first pass

"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from cinnabar.engine_cpp import StaticData, load_teams, parse_team, play_battle
from cinnabar.policy import SmartHeuristicPolicy, StallerPolicy
from cinnabar.search import play_search_battle, selfplay_search_battle
from cinnabar.state import ActionType
from ladder import _load_net

import cinnabar_engine as ce  # noqa: E402

DEFAULT_SWAP_IN = "Jynx:Lovely Kiss,Blizzard,Psychic,Rest"


def parse_swap_in(spec: str) -> tuple[str, list[str]]:
    """'Species:Move1,Move2,...' -> (species, [moves]). Lets follow-up variants (a no-sleeper
    control, Gengar, ...) run without code edits."""
    species, _, moves = spec.partition(":")
    mvs = [m.strip() for m in moves.split(",") if m.strip()]
    if not species.strip() or not mvs:
        raise SystemExit(f"bad --swap-in {spec!r}; expected 'Species:Move1,Move2,...'")
    return species.strip(), mvs[:4]


def _p1_score(r) -> float:
    if r in (ce.Result.Tie, ce.Result.Ongoing):
        return 0.5
    return 1.0 if r == ce.Result.P1Win else 0.0


class SleeperTracker:
    """Per-battle behaviour log for one team slot (the sleeper), fed by the play_battle/
    play_search_battle `observer` hook (called pre-step each turn with (battle, s1, a1)).

    Omniscient on purpose (reads full team_state both sides) — it's a diagnostic, not a player.
    """

    def __init__(self, slot: int) -> None:
        self.slot = slot
        # Aggregates across battles.
        self.battles = 0
        self.turns_active = 0
        self.sleep_clicks = 0
        self.clause_wasted_clicks = 0   # sleep clicked while another unfainted foe already slept
        self.sleeps_landed = 0
        self.sleep_targets: Counter[str] = Counter()
        self.explosions = 0
        self.expl_own_hp = 0.0          # sleeper's HP when it clicked Explosion (sum; avg at print)
        self.expl_foe_hp = 0.0          # foe's HP when Explosion was clicked
        self.expl_targets: Counter[str] = Counter()
        self.fainted = 0                # battles the sleeper ended fainted
        self.battles_slept_someone = 0  # battles with >=1 landed sleep
        self.battles_exploded = 0
        self.kos_while_active = 0       # foe faints on turns the sleeper was on the field (proxy)
        self.damage_while_active = 0.0  # foe HP-fraction lost on those turns (proxy: incl. residuals)
        # Per-battle state.
        self._prev_ots = None
        self._prev_active = False
        self._pending_sleep = False
        self._battle_landed = False
        self._battle_exploded = False

    def start_battle(self) -> None:
        self.battles += 1
        self._prev_ots = None
        self._prev_active = False
        self._pending_sleep = False
        self._battle_landed = False
        self._battle_exploded = False

    def _settle(self, ots) -> None:
        """Compare the foe side vs the last observation; credit the interval to the sleeper if it
        was active then."""
        if self._prev_ots is not None:
            if self._pending_sleep:
                # Did anyone on the foe side transition into SLP? (Attribution caveat: a foe
                # Rest-ing on the exact turn we clicked sleep would false-positive — rare, accepted.)
                for prev, cur in zip(self._prev_ots, ots):
                    if cur[2] == "SLP" and prev[2] != "SLP" and not cur[3]:
                        self.sleeps_landed += 1
                        self.sleep_targets[cur[0]] += 1
                        self._battle_landed = True
                        break
            if self._prev_active:
                self.kos_while_active += sum(1 for p, c in zip(self._prev_ots, ots) if c[3] and not p[3])
                self.damage_while_active += sum(max(0.0, p[1] - c[1]) for p, c in zip(self._prev_ots, ots))
        self._pending_sleep = False

    def __call__(self, battle, s1, a1) -> None:
        ts = battle.team_state(0)
        ots = battle.team_state(1)
        self._settle(ots)
        active = bool(ts[self.slot][4]) and not ts[self.slot][3]
        if active:
            self.turns_active += 1
            if a1.type == ActionType.MOVE:
                if getattr(a1, "effect_status", "") == "SLP":
                    self.sleep_clicks += 1
                    self._pending_sleep = True
                    if any(e[2] == "SLP" and not e[3] for e in ots):
                        self.clause_wasted_clicks += 1
                if getattr(a1, "self_destruct", False):
                    self.explosions += 1
                    self._battle_exploded = True
                    self.expl_own_hp += ts[self.slot][1]
                    foe = next((e for e in ots if e[4]), None)
                    if foe is not None:
                        self.expl_foe_hp += foe[1]
                        self.expl_targets[foe[0]] += 1
        self._prev_ots = ots
        self._prev_active = active

    def finish_battle(self, battle) -> None:
        self._settle(battle.team_state(1))
        if battle.team_state(0)[self.slot][3]:
            self.fainted += 1
        if self._battle_landed:
            self.battles_slept_someone += 1
        if self._battle_exploded:
            self.battles_exploded += 1

    def report(self, name: str) -> None:
        n = max(self.battles, 1)
        print(f"    {name} behaviour ({self.battles} battles):")
        print(f"      turns on field      {self.turns_active / n:6.1f} /battle")
        print(f"      sleep clicks        {self.sleep_clicks / n:6.2f} /battle   "
              f"(clause-wasted: {self.clause_wasted_clicks}/{self.sleep_clicks or 1})")
        print(f"      sleeps landed       {self.sleeps_landed / n:6.2f} /battle   "
              f"(slept someone in {self.battles_slept_someone / n * 100:.0f}% of battles)")
        if self.sleep_targets:
            tgts = ", ".join(f"{s}×{c}" for s, c in self.sleep_targets.most_common(6))
            print(f"      sleep targets       {tgts}")
        if self.explosions:
            print(f"      Explosion clicks    {self.explosions / n:6.2f} /battle   "
                  f"(exploded in {self.battles_exploded / n * 100:.0f}% of battles; "
                  f"avg own HP {self.expl_own_hp / self.explosions * 100:.0f}%, "
                  f"avg foe HP {self.expl_foe_hp / self.explosions * 100:.0f}%)")
            tgts = ", ".join(f"{s}×{c}" for s, c in self.expl_targets.most_common(6))
            print(f"      Explosion targets   {tgts}")
        print(f"      KOs while active    {self.kos_while_active / n:6.2f} /battle (proxy)")
        print(f"      foe dmg while active{self.damage_while_active / n:6.2f} HP-fracs/battle (proxy)")
        print(f"      ended fainted       {self.fainted / n * 100:5.0f}%")


def swap_slot(team, out_species: str, repl) -> tuple[list, int]:
    """Return (team with `out_species`'s slot replaced by `repl`, slot index)."""
    slot = next((i for i, (sp, _) in enumerate(team) if sp == out_species), None)
    if slot is None:
        raise SystemExit(f"{out_species} not found in the base team")
    swapped = list(team)
    swapped[slot] = (repl[0], list(repl[1]))
    return swapped, slot


def main() -> None:
    ap = argparse.ArgumentParser(description="Eggy-vs-Jynx sleeper A/B with behaviour logging.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent / "probe_teams" / "strong.txt"),
                    help="base team file; must contain the --swap-out species")
    ap.add_argument("--swap-out", default="Exeggutor")
    ap.add_argument("--swap-in", default=DEFAULT_SWAP_IN,
                    help="'Species:Move1,Move2,...' replacing --swap-out in variant B "
                         "(e.g. a no-sleeper control: 'Lapras:Blizzard,Thunderbolt,Body Slam,Sing' — pick the set)")
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--pilots", default="raw,search",
                    help="comma-set of raw,search,selfsearch (selfsearch = both sides search-piloted "
                         "— the EXACT evolve_teams fitness judge; --opponent is ignored for it)")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--opponent", choices=["smart", "staller"], default="smart")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)

    swap_in = parse_swap_in(a.swap_in)
    base = parse_team(Path(a.base).read_text())
    team_eggy, slot = swap_slot(base, a.swap_out, next((sp, mv) for sp, mv in base if sp == a.swap_out))
    team_jynx, _ = swap_slot(base, a.swap_out, swap_in)
    opp_teams = load_teams(a.teams_dir)
    if not opp_teams:
        raise SystemExit(f"no teams in {a.teams_dir}")

    net_policy = _load_net(a.ckpt, a.hidden, a.device, 1)
    net = net_policy.net
    opp = SmartHeuristicPolicy() if a.opponent == "smart" else StallerPolicy()
    opp_model = SmartHeuristicPolicy()  # the search pilot's assumed opponent

    # One fixed matchup list — every (pilot, variant) cell plays the identical battles.
    pick = random.Random(a.seed)
    matchups = [(pick.choice(opp_teams), 1000 + i) for i in range(a.battles)]
    pilots = [p.strip() for p in a.pilots.split(",") if p.strip()]
    variants = [(f"{a.swap_out} team", team_eggy), (f"{swap_in[0]} team", team_jynx)]

    print(f"\nSleeper A/B — {Path(a.ckpt).name}, vs {a.opponent}, {a.battles} paired battles, "
          f"clauses {'on' if a.clauses else 'off'}")
    print(f"  base: {Path(a.base).name}; slot {slot}: {a.swap_out} vs {swap_in[0]} "
          f"({'/'.join(swap_in[1])})\n")

    results: dict[tuple[str, str], float] = {}
    for pilot in pilots:
        for vname, team in variants:
            tracker = SleeperTracker(slot)
            wins = 0.0
            for opp_team, seed in matchups:
                tracker.start_battle()
                if pilot == "raw":
                    bat = play_battle(net_policy, opp, team, opp_team, static, seed,
                                      tag=f"pe{seed}", turn_limit=a.turn_limit, clauses=a.clauses,
                                      observer=tracker)
                elif pilot == "selfsearch":  # the evolve_teams fitness judge: search on BOTH sides
                    bat = selfplay_search_battle(net, opp_model, team, opp_team, static, seed,
                                                 clauses=a.clauses, turn_limit=a.turn_limit,
                                                 device=a.device, rollouts=a.rollouts,
                                                 observer=tracker)
                else:
                    bat = play_search_battle(net, opp, opp_model, team, opp_team, static, seed,
                                             tag=f"pe{seed}", turn_limit=a.turn_limit,
                                             clauses=a.clauses, device=a.device,
                                             rollouts=a.rollouts, observer=tracker)
                tracker.finish_battle(bat)
                wins += _p1_score(bat.result())
            wr = wins / a.battles
            results[(pilot, vname)] = wr
            print(f"  {pilot:6s} pilot, {vname:16s} win% {wr * 100:5.1f}")
            tracker.report(team[slot][0])
            print()

    for pilot in pilots:
        eggy = results.get((pilot, variants[0][0]))
        jynx = results.get((pilot, variants[1][0]))
        if eggy is not None and jynx is not None:
            print(f"  {pilot:6s} pilot: {swap_in[0]} - {a.swap_out} = {(jynx - eggy) * 100:+.1f}%  "
                  f"(positive = the pilot extracts more from {swap_in[0]})")


if __name__ == "__main__":
    main()
