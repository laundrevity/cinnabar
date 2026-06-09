"""Policies: the decision-making core.

A `Policy` maps a `BattleState` to an `Action`. This is the seam every future
agent plugs into — heuristic baselines (Phase 1), RL agents (Phase 2), self-play
(Phase 3) — all implement this one method. No poke-env here.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from .state import Action, ActionType, BattleState


class Policy(ABC):
    """Maps a battle state to a chosen action."""

    @abstractmethod
    def select_action(self, state: BattleState) -> Action:
        ...


class RandomPolicy(Policy):
    """Picks uniformly at random among the legal actions. The Phase 0 baseline."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: BattleState) -> Action:
        if not state.available_actions:
            raise ValueError("RandomPolicy called with no available actions")
        return self._rng.choice(state.available_actions)


class MaxDamagePolicy(Policy):
    """Type-aware max-damage baseline (Phase 1).

    Scores each available move as::

        estimate = base_power * type_multiplier * STAB

    where STAB is 1.5 when the move's type matches one of the active Pokémon's
    types, and plays the highest-scoring move. Status / ineffective moves score 0
    and are avoided. On a turn with no damaging move (e.g. a forced switch) it
    switches; if it can't, it falls back to any legal action.

    Deliberately simple: no switch evaluation, no opponent prediction, no PP or
    accuracy weighting yet. Its only job is to be a fixed, clearly-better-than-
    random yardstick for everything that comes after.
    """

    STAB_MULTIPLIER = 1.5

    def select_action(self, state: BattleState) -> Action:
        if not state.available_actions:
            raise ValueError("MaxDamagePolicy called with no available actions")

        moves = [a for a in state.available_actions if a.type == ActionType.MOVE]
        if moves:
            best = max(moves, key=lambda a: self._estimate(a, state))
            if self._estimate(best, state) > 0:
                return best

        switches = [a for a in state.available_actions if a.type == ActionType.SWITCH]
        if switches:
            return switches[0]

        return state.available_actions[0]

    def _estimate(self, action: Action, state: BattleState) -> float:
        base = action.base_power or 0.0
        multiplier = action.type_multiplier if action.type_multiplier is not None else 1.0
        stab = 1.0
        active = state.active
        if active and action.move_type and action.move_type in active.types:
            stab = self.STAB_MULTIPLIER
        return base * multiplier * stab


class SmartHeuristicPolicy(Policy):
    """Max-damage PLUS the levers max-damage ignores: sleep / paralysis status, healing,
    and pivoting off a bad matchup.

    Its purpose is diagnostic. Max-damage is a *fixed, exploitable* yardstick; "win% vs
    max-damage" turned out not to track real strength (mirror eval). This policy measures
    how much skill room exists *above* max-damage on a given team set, using only mechanics
    the engine actually models (sleep, paralysis, Recover/Soft-Boiled/Rest, switching). If
    it can't clear ~55-60% mirrored vs max-damage, the room isn't there as modeled (the
    bottleneck is engine breadth); if it can, the room exists and the RL agent is missing it.
    """

    STAB = 1.5
    SLEEP_MOVES = {"sleeppowder", "hypnosis", "lovelykiss", "spore", "sing"}
    PARA_MOVES = {"thunderwave", "stunspore", "glare"}
    HEAL_MOVES = {"recover", "softboiled", "rest"}

    def select_action(self, state: BattleState) -> Action:
        acts = state.available_actions
        if not acts:
            raise ValueError("SmartHeuristicPolicy called with no available actions")
        moves = [a for a in acts if a.type == ActionType.MOVE]
        switches = [a for a in acts if a.type == ActionType.SWITCH]

        if not moves:  # forced switch / no usable move: safest, healthiest body
            return self._best_switch(switches) or acts[0]

        active, opp = state.active, state.opponent_active
        best = max(moves, key=lambda a: self._dmg(a, active))
        best_val = self._dmg(best, active)

        # Hyper Beam wastes the next turn recharging unless it KOs — a core Gen 1 skill that
        # max-damage ignores. If the top pick is Hyper Beam and it won't KO, prefer the best
        # non-recharge move instead. (Only meaningful once the engine models the recharge.)
        likely_ko = opp is not None and opp.hp_fraction < 0.35 and best_val >= 80
        if best.move_id == "hyperbeam" and not likely_ko:
            alt = [m for m in moves if m.move_id != "hyperbeam"]
            alt_best = max(alt, key=lambda a: self._dmg(a, active)) if alt else None
            # Only skip the recharge if a near-as-strong move exists — don't give up a much
            # bigger hit just to dodge the recharge turn.
            if alt_best is not None and self._dmg(alt_best, active) >= 0.6 * best_val:
                best, best_val = alt_best, self._dmg(alt_best, active)
                likely_ko = opp is not None and opp.hp_fraction < 0.35 and best_val >= 80

        # If we likely KO, just hit.
        if likely_ko:
            return best

        # Sleep a healthy, un-statused foe — the strongest tempo move in Gen 1.
        if opp is not None and not opp.status and opp.hp_fraction > 0.55:
            slp = self._pick(moves, self.SLEEP_MOVES, min_acc=0.6)
            if slp is not None:
                return slp

        # Paralyze a healthy, faster, un-statused foe to flip speed control. Skip if immune.
        if opp is not None and not opp.status and active is not None \
                and (opp.speed or 0) > (active.speed or 0):
            par = self._pick(moves, self.PARA_MOVES, need_effective=True)
            if par is not None:
                return par

        # Heal when low (and not already about to KO the foe).
        if active is not None and active.hp_fraction < 0.45:
            heal = self._pick(moves, self.HEAL_MOVES)
            if heal is not None:
                return heal

        # Walled (best hit is resisted and weak) and a safer body exists: pivot.
        base = best.base_power or 0.0
        if base and best_val <= 0.5 * base and best_val < 60 and switches:
            sw = self._best_switch(switches)
            if sw is not None and (sw.incoming_multiplier or 1.0) < 1.0:
                return sw

        return best

    def _dmg(self, a: Action, active) -> float:
        base = a.base_power or 0.0
        if not base and a.fixed_damage:
            return float(a.fixed_damage)  # Seismic Toss / Night Shade: flat ~100
        mult = a.type_multiplier if a.type_multiplier is not None else 1.0
        stab = self.STAB if (active and a.move_type and a.move_type in active.types) else 1.0
        acc = a.accuracy if a.accuracy is not None else 1.0
        return base * mult * stab * acc

    def _pick(self, moves, ids, *, min_acc: float = 0.0, need_effective: bool = False):
        for a in moves:
            if a.move_id in ids and (a.accuracy if a.accuracy is not None else 1.0) >= min_acc:
                if need_effective and a.type_multiplier == 0:
                    continue
                return a
        return None

    def _best_switch(self, switches):
        if not switches:
            return None
        return min(switches, key=lambda s: ((s.incoming_multiplier or 1.0),
                                            -(s.target_hp_fraction or 0.0)))


class StallerPolicy(SmartHeuristicPolicy):
    """A patient paralysis + recovery staller — the human style that beats the RL agent but that
    nothing in its training reproduces (the league is past selves; the heuristic just attacks).

    It spreads paralysis proactively (not only when slower), recovers to stay healthy rather than
    only in emergencies, grinds with its best hit, pivots off a losing matchup, and respects Sleep
    Clause (never re-sleeps). Purpose: a diagnostic opponent to test whether the net escapes a losing
    attrition fight — and, if it folds, a training opponent that finally punishes the blind spot.
    """

    RECOVER_BELOW = 0.65

    def select_action(self, state: BattleState) -> Action:
        acts = state.available_actions
        if not acts:
            raise ValueError("StallerPolicy called with no available actions")
        moves = [a for a in acts if a.type == ActionType.MOVE]
        switches = [a for a in acts if a.type == ActionType.SWITCH]
        if not moves:
            return self._best_switch(switches) or acts[0]

        active, opp = state.active, state.opponent_active
        best = max(moves, key=lambda a: self._dmg(a, active))

        # Close out only when a real KO is on the table.
        if opp is not None and opp.hp_fraction < 0.30 and self._dmg(best, active) >= 70:
            return best

        # Sleep a healthy un-statused foe, but never re-sleep into Sleep Clause.
        foe_asleep = any(m.status == "SLP" for m in state.opponent_team)
        if opp is not None and not opp.status and opp.hp_fraction > 0.5 and not foe_asleep:
            slp = self._pick(moves, self.SLEEP_MOVES, min_acc=0.55)
            if slp is not None:
                return slp

        # Spread paralysis proactively — the core staller lever (SmartHeuristic only paralyses when slower).
        if opp is not None and not opp.status:
            par = self._pick(moves, self.PARA_MOVES, need_effective=True)
            if par is not None:
                return par

        # Recover early to win the long game, not just to avoid dying.
        if active is not None and active.hp_fraction < self.RECOVER_BELOW:
            heal = self._pick(moves, self.HEAL_MOVES)
            if heal is not None:
                return heal

        # Pivot off a losing matchup (best hit resisted/weak) to a safer body.
        base = best.base_power or 0.0
        if base and self._dmg(best, active) <= 0.5 * base and switches:
            sw = self._best_switch(switches)
            if sw is not None and (sw.incoming_multiplier or 1.0) < 1.0:
                return sw

        return best
