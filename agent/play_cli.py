"""Play against the agent in the terminal — no browser, no Showdown server.

The whole game runs on the in-process C++ engine: you are P1 choosing from a numbered
menu, the agent is P2 piloted exactly like its strongest configurations (raw policy,
decision-time search, or a heuristic). Information is partial BOTH ways, like the real
game: you see the opponent's active mon and whatever its side has revealed; the agent
sees the same about you. Plain text only.

    cd agent
    uv run python play_cli.py --ckpt models_fast/pg_best.pt                    # search pilot
    uv run python play_cli.py --ckpt models_fast/pg_best.pt --pilot raw        # greedy policy
    uv run python play_cli.py --ckpt models_fast/pg_best.pt --value-ckpt value_fast/value_best.pt \
        --minimax --opp-top-k 3 --opp-temp 0.35                                # the heavy config
    uv run python play_cli.py --pilot staller                                  # no checkpoint needed

In battle: type the action number, `t` for full team detail, `q` to forfeit.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

from cinnabar.encoding import encode_global
from cinnabar.engine_cpp import Reveal, StaticData, build_state, load_teams, parse_team, reveal_move
from cinnabar.policy import SmartHeuristicPolicy, StallerPolicy
from cinnabar.search import _will_fail
from cinnabar.state import ActionType
from ladder import PolicyPilot, SearchPilot, _load_net, _load_value
from watch_engine import _deltas

import cinnabar_engine as ce  # noqa: E402

BOOST_NAMES = ("atk", "def", "spc", "spe")
VOL_TAGS = (("must_recharge", "recharging"), ("confused", "confused"), ("reflect", "reflect"),
            ("light_screen", "light screen"), ("leech_seeded", "leech seed"),
            ("disabled", "disabled"), ("toxic", "toxic"))


def _active_line(mon) -> str:
    """'Snorlax 78% par +2 spc, reflect' — the active mon with boosts and volatiles."""
    if mon is None:
        return "---"
    s = f"{mon.species} {mon.hp_fraction * 100:.0f}%"
    if mon.status:
        s += f" {mon.status.lower()}"
    boosts = [f"{v:+d} {BOOST_NAMES[i]}" for i, v in enumerate(mon.boosts) if v]
    tags = [label for attr, label in VOL_TAGS if getattr(mon, attr, False)]
    if mon.sleep_turns:
        tags.append(f"asleep ~{mon.sleep_turns}t")
    extra = boosts + tags
    if extra:
        s += "  [" + ", ".join(extra) + "]"
    return s


def _eff(mult: float | None) -> str:
    if mult is None or mult == 1.0:
        return ""
    return f" x{mult:g}"


def _action_line(a, state) -> str:
    """One menu row. Moves show type/power/accuracy/effectiveness + effect tags; switches
    show what you'd switch into and how hard the foe's typing hits it."""
    if a.type == ActionType.SWITCH:
        s = f"switch {a.species:12s} {(a.target_hp_fraction or 0) * 100:3.0f}%"
        if a.target_statused:
            s += " (statused)"
        inc = a.incoming_multiplier
        if inc is not None and inc != 1.0:
            s += f"  [takes x{inc:g}]"
        return s
    power = (f"={a.fixed_damage:.0f}hp" if a.fixed_damage
             else f"{a.base_power:3.0f}" if a.base_power else "  -")
    acc = f"{(a.accuracy if a.accuracy is not None else 1.0) * 100:3.0f}%"
    s = f"{a.label:14s} {a.move_type or '':8s} {power}  acc{acc}"
    if a.base_power or a.fixed_damage:
        s += _eff(a.type_multiplier)
    tags = []
    if a.effect_status:
        chance = "" if a.effect_chance >= 0.999 else f" {a.effect_chance * 100:.0f}%"
        tags.append(a.effect_status.lower() + chance)
    if a.heals:
        tags.append("heal")
    if a.boosts_self:
        tags.append("boost")
    if a.lowers_foe:
        tags.append("lower")
    if a.recharge:
        tags.append("recharge")
    if a.self_destruct:
        tags.append("faints user")
    if _will_fail(a, state):
        tags.append("WILL FAIL")
    if tags:
        s += "  [" + ", ".join(tags) + "]"
    return s


def _print_board(state, n_foe_total: int, turn: int) -> None:
    print(f"\n--- turn {turn} ---")
    print(f"  you: {_active_line(state.active)}")
    print(f"  foe: {_active_line(state.opponent_active)}")
    bench = [m for m in state.team if not m.active]
    print("  your bench: " + (", ".join(
        f"{m.species} {m.hp_fraction * 100:.0f}%{' ' + m.status.lower() if m.status else ''}"
        + (" KO" if m.fainted else "") for m in bench) or "-"))
    revealed = [m for m in state.opponent_team if not m.active]
    hidden = n_foe_total - len(state.opponent_team)
    foe_bits = [f"{m.species} {m.hp_fraction * 100:.0f}%{' ' + m.status.lower() if m.status else ''}"
                + (" KO" if m.fainted else "") for m in revealed]
    if hidden > 0:
        foe_bits.append(f"({hidden} unrevealed)")
    print("  foe revealed: " + (", ".join(foe_bits) or f"({hidden} unrevealed)"))
    if state.opponent_revealed_moves:
        print("  foe active has shown: " + ", ".join(a.label for a in state.opponent_revealed_moves))


def _menu(state) -> None:
    for i, a in enumerate(state.available_actions, 1):
        print(f"  {i}. {_action_line(a, state)}")


def _ask(state):
    """Prompt until a legal action (or None = forfeit). `t` re-prints the board, enter the menu."""
    n = len(state.available_actions)
    while True:
        try:
            raw = input("> ").strip().lower()
        except EOFError:
            return None
        if raw in ("q", "quit", "forfeit"):
            return None
        if raw in ("", "?", "h", "help"):
            _menu(state)
            continue
        if raw in ("t", "team"):
            for m in state.team:
                mark = "*" if m.active else " "
                print(f"  {mark} {m.species:12s} {m.hp_fraction * 100:3.0f}% "
                      f"{m.status.lower() if m.status else ''}{' KO' if m.fainted else ''}")
            continue
        if raw.isdigit() and 1 <= int(raw) <= n:
            return state.available_actions[int(raw) - 1]
        print(f"  (1-{n}, enter = menu, t = team, q = forfeit)")


def play_one(my_team, opp_team, pilot, static, seed, *, clauses, turn_limit, device,
             eval_net=None) -> None:
    spec1 = [(s, list(m)) for s, m in my_team]
    spec2 = [(s, list(m)) for s, m in opp_team]
    battle = ce.make_battle(spec1, spec2, seed)
    if clauses:
        battle.set_clauses(True)
    r1, r2 = Reveal(), Reveal()
    rng = random.Random(seed * 31 + 1)  # the agent's search dice
    print(f"\nyour team: {', '.join(s for s, _ in spec1)}")
    turns = 0
    while battle.result() == ce.Result.Ongoing and turns < turn_limit:
        turns += 1
        s1 = build_state(battle, 0, spec1, static, "cli", reveal=r1, opp_team=spec2)
        s2 = build_state(battle, 1, spec2, static, "cli_o", reveal=r2, opp_team=spec1)

        agent_replacing = battle.must_switch(1) and not battle.must_switch(0)
        if agent_replacing:
            a1 = s1.available_actions[0]  # the engine ignores the waiting side's choice
        elif len(s1.available_actions) == 1:
            a1 = s1.available_actions[0]  # recharge / trap-locked / last mon: nothing to decide
            print(f"\n--- turn {turns} ---  (auto: {a1.label})")
        else:
            _print_board(s1, n_foe_total=len(spec2), turn=turns)
            if s1.force_switch:
                print("  (your active fainted — choose a replacement)")
            _menu(s1)
            a1 = _ask(s1)
            if a1 is None:
                print("\n  >>> you forfeit.")
                return
        a2 = s2.available_actions[pilot.choose(battle, 1, s2, r2, spec2, spec1, static, rng)]

        pre1, pre2 = battle.team_state(0), battle.team_state(1)
        reveal_move(r1, s2, a2)
        reveal_move(r2, s1, a1)
        battle.step(battle.choices(0)[a1.index], battle.choices(1)[a2.index])

        if not agent_replacing:
            print(f"\n  you: {a1.label}   |   foe: {a2.label}")
        for line in _deltas("you:", pre1, battle.team_state(0)) \
                + _deltas("foe:", pre2, battle.team_state(1)):
            print(f"    {line}")
        if eval_net is not None and battle.result() == ce.Result.Ongoing:
            s2v = build_state(battle, 1, spec2, static, "cli_v", reveal=r2, opp_team=spec1)
            with torch.no_grad():
                v = float(eval_net.value(torch.tensor(encode_global(s2v), dtype=torch.float32,
                                                      device=device)))
            print(f"    (agent thinks it wins with p={v:.2f})")

    res = battle.result()
    msg = {ce.Result.P1Win: "you win!", ce.Result.P2Win: "the agent wins.",
           ce.Result.Tie: "tie."}.get(res, f"no result in {turn_limit} turns.")
    print(f"\n  >>> {msg} (turn {turns})")


def _pick_team(teams, names, prompt: str):
    """Interactive team pick when stdin is a terminal; random otherwise / on enter."""
    if not sys.stdin.isatty():
        i = random.randrange(len(teams))
        return teams[i], names[i]
    print(f"\n{prompt}")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    try:
        raw = input("team (enter = random) > ").strip()
    except EOFError:
        raw = ""
    i = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(teams) else random.randrange(len(teams))
    return teams[i], names[i]


def main() -> None:
    ap = argparse.ArgumentParser(description="Play the agent in the terminal (C++ engine).")
    ap.add_argument("--ckpt", default=None, help="net checkpoint (needed for raw/search pilots)")
    # raw is the measured default: search's lift is matchup-dependent for current nets
    # (+8.5% vs the attacker but -10% vs the staller — and patient humans play staller-shaped).
    ap.add_argument("--pilot", choices=["search", "raw", "heuristic", "staller"], default="raw")
    ap.add_argument("--value-ckpt", default=None,
                    help="calibrated ValueNet as the search leaf (HybridNet, like search_eval)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--minimax", action="store_true")
    ap.add_argument("--opp-top-k", type=int, default=3)
    ap.add_argument("--opp-temp", type=float, default=0.0)
    ap.add_argument("--team", default=None, help="your team .txt (Showdown export); default: pick/list")
    ap.add_argument("--opp-team", default=None, help="agent's team .txt; default: random from the pool")
    ap.add_argument("--teams-dir", default=str(Path(__file__).resolve().parent.parent / "teams"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-clauses", dest="clauses", action="store_false",
                    help="disable Sleep/Freeze Clause (default: on, the real format)")
    ap.add_argument("--show-eval", action="store_true",
                    help="print the agent's estimated win probability each turn")
    ap.add_argument("--turn-limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else random.randrange(1, 10**9)
    random.seed(seed)
    static = StaticData(1)

    net_policy = None
    if a.pilot in ("raw", "search"):
        if not a.ckpt:
            raise SystemExit(f"--pilot {a.pilot} needs --ckpt")
        net_policy = _load_net(a.ckpt, a.hidden, a.device, 1)
    if a.pilot == "search":
        net = net_policy.net
        if a.value_ckpt:
            from cinnabar.rl.net import HybridNet
            net = HybridNet(net, _load_value(a.value_ckpt, a.hidden, a.device))
        pilot = SearchPilot(net, rollouts=a.rollouts, top_k=a.top_k, minimax=a.minimax,
                            opp_top_k=a.opp_top_k, opp_temp=a.opp_temp, device=a.device)
        eval_net = net
    elif a.pilot == "raw":
        pilot, eval_net = PolicyPilot(net_policy), net_policy.net
    else:
        pilot = PolicyPilot(SmartHeuristicPolicy() if a.pilot == "heuristic" else StallerPolicy())
        eval_net = None
    if not a.show_eval:
        eval_net = None

    files = sorted(Path(a.teams_dir).glob("*.txt"))
    pool = load_teams(a.teams_dir)
    names = [f.stem for f in files]
    if not pool and not (a.team and a.opp_team):
        raise SystemExit(f"no teams in {a.teams_dir}")

    print(f"pilot: {a.pilot} | clauses {'on' if a.clauses else 'off'} | seed {seed}")
    game = 0
    while True:
        if a.team:
            my_team, my_name = parse_team(Path(a.team).read_text()), Path(a.team).stem
        else:
            my_team, my_name = _pick_team(pool, names, "pick your team:")
        if a.opp_team:
            opp_team = parse_team(Path(a.opp_team).read_text())
        else:
            opp_team = random.choice(pool)
        print(f"\n=== game {game + 1}: {my_name} vs ??? ===")
        play_one(my_team, opp_team, pilot, static, seed + game * 7919,
                 clauses=a.clauses, turn_limit=a.turn_limit, device=a.device, eval_net=eval_net)
        game += 1
        if not sys.stdin.isatty():
            return
        try:
            if input("\nplay again? (y/n) > ").strip().lower() not in ("y", "yes", ""):
                return
        except EOFError:
            return


if __name__ == "__main__":
    main()
