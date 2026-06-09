"""Diagnose the agent's play — is it the impatient switch-looper the evolved meta implies?

Three measurements for a checkpoint, all vs the smart heuristic on a team pool:

  1. Voluntary switch rate (switches / non-forced decisions), printed next to the heuristic's rate
     as a sane baseline. The ping-pong shows up here as a switch rate well above the heuristic's.
  2. Average game length + share of games that hit the turn limit (stall-loops that never resolve).
  3. An A/B on the SAME hand-built defensive team: net-as-pilot vs heuristic-as-pilot, both facing
     the heuristic on the pool. If the net wins less with the identical team, it's the worse pilot of
     patient defense — the mechanism behind evolution cutting Chansey / Snorlax / Exeggutor.

The two probe teams are diagnostic instruments, not training priors — they only measure the agent.

    cd agent
    uv run python diagnose_policy.py --ckpt models_clauses/pg_best.pt --battles 300 --clauses
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from cinnabar.engine_cpp import StaticData, parse_team, play_battle
from cinnabar.policy import Policy, SmartHeuristicPolicy, StallerPolicy
from cinnabar.state import ActionType
from ladder import _load_net

import cinnabar_engine as ce  # noqa: E402

# Diagnostic probe: a textbook patient/defensive team (walls + recovery + status). Used only to
# MEASURE pilot skill on defense — it is never fed to training or evolution.
DEFENSIVE_TEAM = """Chansey
- Ice Beam
- Thunder Wave
- Soft-Boiled
- Counter

Snorlax
- Body Slam
- Reflect
- Rest
- Earthquake

Slowbro
- Surf
- Amnesia
- Rest
- Thunder Wave

Starmie
- Surf
- Recover
- Thunder Wave
- Blizzard

Alakazam
- Psychic
- Recover
- Thunder Wave
- Seismic Toss

Exeggutor
- Sleep Powder
- Psychic
- Stun Spore
- Mega Drain
"""


class Counting(Policy):
    """Wrap a policy and tally its action types (forced switches on a faint don't count as choices)."""

    def __init__(self, inner: Policy) -> None:
        self.inner = inner
        self.moves = 0
        self.switches = 0
        self.forced = 0
        # Sleep-clause discipline: of the turns where a foe is ALREADY asleep and a sleep move is
        # available (so sleeping again would just fail under the clause), how often does it sleep?
        self.sleep_tempt = 0
        self.sleep_reslept = 0
        # Switch-LOOP tail metric (a watched browser game showed an 8-turn resist-swap ping-pong
        # that the average switch rate dilutes away): longest run of consecutive voluntary
        # switches, and how many battles contain a run >= 5.
        self.max_streak = 0
        self.loopy_battles = 0
        self._tag = None
        self._streak = 0
        self._battle_max = 0

    def _new_battle(self, tag) -> None:
        if self._battle_max >= 5:
            self.loopy_battles += 1
        self._tag = tag
        self._streak = 0
        self._battle_max = 0

    def flush(self) -> None:
        """Count the final battle's streak (call once after the last battle)."""
        self._new_battle(None)

    def select_action(self, state):
        if state.battle_tag != self._tag:
            self._new_battle(state.battle_tag)
        foe_asleep = any(getattr(m, "status", None) == "SLP" for m in getattr(state, "opponent_team", []))
        sleep_avail = any(getattr(x, "effect_status", "") == "SLP" for x in state.available_actions)
        a = self.inner.select_action(state)
        if foe_asleep and sleep_avail:
            self.sleep_tempt += 1
            if getattr(a, "effect_status", "") == "SLP":
                self.sleep_reslept += 1
        if getattr(state, "force_switch", False):
            self.forced += 1
            self._streak = 0
        elif a.type == ActionType.SWITCH:
            self.switches += 1
            self._streak += 1
            self._battle_max = max(self._battle_max, self._streak)
            self.max_streak = max(self.max_streak, self._streak)
        else:
            self.moves += 1
            self._streak = 0
        return a

    @property
    def switch_rate(self) -> float:
        d = self.switches + self.moves
        return self.switches / d if d else 0.0

    @property
    def reslept_rate(self) -> float:
        return self.sleep_reslept / self.sleep_tempt if self.sleep_tempt else 0.0


def _team_score(r, lead) -> float:
    if r in (ce.Result.Tie, ce.Result.Ongoing):
        return 0.5
    return 1.0 if (r == ce.Result.P1Win) == lead else 0.0


def behavior(p1, p2, teams, n, static, clauses, turn_limit, seed0,
             counter=None, names=None, loopy=None):
    """p1's win-rate, avg turns, and turn-limit (stall) rate over n battles, leads alternated.
    If `counter` (the Counting wrapper around p1) + `names` (id(team) -> file stem) + `loopy`
    (output list) are given, battles where p1 ran a 5+ voluntary-switch streak are recorded as
    (seed, p1-team-name, p2-team-name, lead, streak) so they can be replayed in watch_engine.py."""
    win = turns = stalls = 0.0
    for i in range(n):
        t1, t2 = random.choice(teams), random.choice(teams)
        lead = i % 2 == 0
        pa, pb = (p1, p2) if lead else (p2, p1)
        s = seed0 + i
        bat = play_battle(pa, pb, t1, t2, static, s, tag=f"d{s}",
                          turn_limit=turn_limit, clauses=clauses)
        r = bat.result()
        turns += bat.turn
        stalls += 1.0 if r == ce.Result.Ongoing else 0.0
        win += _team_score(r, lead)
        if counter is not None and loopy is not None and counter._battle_max >= 5:
            loopy.append((s, names[id(t1)], names[id(t2)], lead, counter._battle_max))
        seed0 += 1
    return win / n, turns / n, stalls / n


def pilot_winrate(pilot, my_team, opp_pilot, opp_teams, n, static, clauses, turn_limit, seed0):
    """Win-rate of `pilot` playing `my_team` vs `opp_pilot` on a fixed opponent-team sequence (so two
    pilots can be compared on identical matchups). Leads alternated."""
    win = 0.0
    for i in range(n):
        opp = opp_teams[i % len(opp_teams)]
        lead = i % 2 == 0
        a, b = (pilot, opp_pilot) if lead else (opp_pilot, pilot)
        t1, t2 = (my_team, opp) if lead else (opp, my_team)
        r = play_battle(a, b, t1, t2, static, seed0 + i, tag=f"p{seed0+i}",
                        turn_limit=turn_limit, clauses=clauses).result()
        win += _team_score(r, lead)
    return win / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the agent's play (switch-loop / defensive skill).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--battles", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clauses", action="store_true")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    static = StaticData(1)
    named = [(p.stem, parse_team(p.read_text())) for p in sorted(Path(a.teams_dir).glob("*.txt"))]
    named = [(nm, t) for nm, t in named if t]
    teams = [t for _, t in named]
    names = {id(t): nm for nm, t in named}
    if not teams:
        raise SystemExit(f"no teams in {a.teams_dir}")
    net = _load_net(a.ckpt, a.hidden, a.device, 1)
    smart = SmartHeuristicPolicy()

    def replay_cmds(loopy, opp_pilot):
        """Ready-to-run watch_engine.py commands reproducing each loopy battle (same teams, seed,
        deterministic pilots — if a replay diverges, the heuristic is using unseeded RNG)."""
        cl = " --clauses" if a.clauses else ""
        for s, t1n, t2n, lead, mx in loopy[:5]:
            p1p, p2p = ("raw", opp_pilot) if lead else (opp_pilot, "raw")
            print(f"      uv run python watch_engine.py --ckpt {a.ckpt} --p1 {a.teams_dir}/{t1n}.txt "
                  f"--p2 {a.teams_dir}/{t2n}.txt --p1-pilot {p1p} --p2-pilot {p2p} --seed {s} "
                  f"--turn-limit {a.turn_limit}{cl}   # streak {mx}")

    # 1+2. switch rate, game length, stall rate — net vs heuristic.
    cnet, csmart = Counting(net), Counting(smart)
    loopy_smart: list = []
    win, avg_turns, stall = behavior(cnet, csmart, teams, a.battles, static, a.clauses, a.turn_limit, 1,
                                     counter=cnet, names=names, loopy=loopy_smart)
    print(f"\nnet ({Path(a.ckpt).name}) vs smart heuristic — {a.battles} battles, "
          f"clauses {'on' if a.clauses else 'off'}\n")
    print(f"  net win%            {win*100:5.1f}")
    print(f"  net switch rate     {cnet.switch_rate*100:5.1f}%   (voluntary switches / decisions)")
    print(f"  smart switch rate   {csmart.switch_rate*100:5.1f}%   (sane baseline)")
    print(f"  avg game length     {avg_turns:5.1f} turns")
    print(f"  hit turn limit      {stall*100:5.1f}%   (stall-loops that never resolve)")
    print(f"  re-slept rate       {cnet.reslept_rate*100:5.1f}%   "
          f"({cnet.sleep_reslept}/{cnet.sleep_tempt} times a foe was already asleep + a sleep move was up)")
    cnet.flush()
    print(f"  switch-loop tail    longest streak {cnet.max_streak}, "
          f"battles with a 5+ streak: {cnet.loopy_battles}/{a.battles}")
    if loopy_smart:
        print("    replay the worst (turn-by-turn transcript):")
        replay_cmds(sorted(loopy_smart, key=lambda x: -x[4]), "heuristic")

    # 3. same defensive team, two pilots, identical opponents.
    def_team = parse_team(DEFENSIVE_TEAM)
    n_ab = max(60, a.battles // 2)
    net_def = pilot_winrate(net, def_team, smart, teams, n_ab, static, a.clauses, a.turn_limit, 10_000)
    smt_def = pilot_winrate(smart, def_team, smart, teams, n_ab, static, a.clauses, a.turn_limit, 10_000)
    # 2b. vs a PATIENT STALLER — the human style nothing in training reproduces. If the net wins far
    # less here than vs the (attacking) heuristic, it can't handle patient paralysis+recovery stall.
    cnet2 = Counting(net)
    win_st, turns_st, stall_st = behavior(cnet2, StallerPolicy(), teams, a.battles, static,
                                          a.clauses, a.turn_limit, 5_000)
    print("\n  vs a patient staller (paralysis + recovery + pivot):")
    print(f"    net win%        {win_st*100:5.1f}%   (vs {win*100:.1f}% against the attacking heuristic)")
    print(f"    net switch rate {cnet2.switch_rate*100:5.1f}%")
    print(f"    avg game length {turns_st:5.1f} turns, hit turn limit {stall_st*100:.1f}%")

    print(f"\n  piloting the SAME defensive team vs smart ({n_ab} battles, identical matchups):")
    print(f"    net pilot       {net_def*100:5.1f}%")
    print(f"    smart pilot     {smt_def*100:5.1f}%")
    gap = (smt_def - net_def) * 100
    print(f"    gap             {gap:+5.1f}%   (positive = the net is the worse pilot of defense)")

    # 4. MIRROR play — net vs net, the configuration the evolution judge and self-play actually run
    # on, and where the switch-loop lives (watched browser game, 2026-06: an 8-turn resist-swap
    # ping-pong that no vs-heuristic probe ever showed). Win% is 50% by construction; the switch
    # rate and game length are the signal.
    cnet3 = Counting(net)
    loopy_mirror: list = []
    _, turns_m, stall_m = behavior(cnet3, net, teams, a.battles, static, a.clauses, a.turn_limit, 20_000,
                                   counter=cnet3, names=names, loopy=loopy_mirror)
    print(f"\n  MIRROR (net vs net — the judge's configuration):")
    print(f"    net switch rate {cnet3.switch_rate*100:5.1f}%   (vs {cnet.switch_rate*100:.1f}% against smart)")
    print(f"    re-slept rate   {cnet3.reslept_rate*100:5.1f}%")
    print(f"    avg game length {turns_m:5.1f} turns, hit turn limit {stall_m*100:.1f}%")
    cnet3.flush()
    print(f"    switch-loop tail  longest streak {cnet3.max_streak}, "
          f"battles with a 5+ streak: {cnet3.loopy_battles}/{a.battles}")
    if loopy_mirror:
        print("    replay the worst (turn-by-turn transcript):")
        replay_cmds(sorted(loopy_mirror, key=lambda x: -x[4]), "raw")


if __name__ == "__main__":
    main()
