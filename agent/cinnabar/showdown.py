"""The poke-env adapter — the *only* module that touches poke-env.

`PolicyPlayer` is a poke-env `Player` that delegates every decision to one of our
`Policy` objects. Each turn it:

  1. builds our `BattleState` from poke-env's `Battle`,
  2. asks the policy for an `Action`,
  3. maps that `Action` back to a Showdown order.

If we ever move off poke-env, this is the file to rewrite.
"""

from __future__ import annotations

from poke_env.player import Player

from .policy import Policy
from .state import Action, ActionType, ActivePokemon, BattleState, TeamMon


class PolicyPlayer(Player):
    def __init__(self, policy: Policy, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._policy = policy
        self._type_charts: dict = {}  # gen -> type chart, lazily cached

    def choose_move(self, battle):
        state, targets = self._build_state(battle)
        if not state.available_actions:
            # No moves/switches we modelled (e.g. odd forced state) — let
            # poke-env pick a guaranteed-valid order.
            return self.choose_random_move(battle)
        action = self._policy.select_action(state)
        return self.create_order(targets[action.index])

    # -- translation: poke-env Battle -> our BattleState -------------------------

    def _build_state(self, battle) -> tuple[BattleState, list]:
        """Returns our BattleState plus a parallel list of poke-env order targets
        (Move/Pokemon objects), indexed identically to `available_actions`."""
        actions: list[Action] = []
        targets: list = []

        for move in battle.available_moves:
            actions.append(
                Action(
                    index=len(actions),
                    type=ActionType.MOVE,
                    label=move.id,
                    move_id=move.id,
                    base_power=move.base_power,
                    move_type=move.type.name if move.type else None,
                    category=move.category.name if move.category else None,
                    accuracy=move.accuracy,
                    type_multiplier=self._type_multiplier(battle, move),
                    fixed_damage=self._fixed_damage(move),
                )
            )
            targets.append(move)

        for mon in battle.available_switches:
            actions.append(
                Action(
                    index=len(actions),
                    type=ActionType.SWITCH,
                    label=f"switch:{mon.species}",
                    species=mon.species,
                    target_hp_fraction=mon.current_hp_fraction,
                    target_statused=mon.status is not None,
                    incoming_multiplier=self._incoming_multiplier(battle, mon),
                )
            )
            targets.append(mon)

        state = BattleState(
            turn=battle.turn,
            active=self._mon_view(battle.active_pokemon),
            opponent_active=self._mon_view(battle.opponent_active_pokemon),
            available_actions=actions,
            force_switch=bool(getattr(battle, "force_switch", False)),
            battle_tag=getattr(battle, "battle_tag", None),
            team=[self._team_mon(m) for m in battle.team.values()],
            opponent_team=[self._team_mon(m) for m in battle.opponent_team.values()],
        )
        return state, targets

    @staticmethod
    def _mon_view(mon) -> ActivePokemon | None:
        if mon is None:
            return None
        return ActivePokemon(
            species=mon.species,
            hp_fraction=mon.current_hp_fraction,
            status=mon.status.name if mon.status else None,
            types=tuple(t.name for t in mon.types if t is not None),
            speed=(getattr(mon, "base_stats", None) or {}).get("spe"),
        )

    def _chart(self, battle):
        """The format's type chart, lazily fetched and cached per generation."""
        gen = getattr(battle, "gen", 1)
        chart = self._type_charts.get(gen)
        if chart is None:
            from poke_env.data import GenData

            chart = GenData.from_gen(gen).type_chart
            self._type_charts[gen] = chart
        return chart

    def _type_multiplier(self, battle, move) -> float | None:
        """Type effectiveness of `move` against the current opponent active.
        Returns None when there's no opponent to compare against (1.0 to policies)."""
        opponent = battle.opponent_active_pokemon
        if opponent is None or move.type is None:
            return None
        return move.type.damage_multiplier(*opponent.types, type_chart=self._chart(battle))

    def _incoming_multiplier(self, battle, mon) -> float | None:
        """Worst-case effectiveness of the opponent active's types against `mon`
        (a switch target) — how dangerous it is to switch this Pokémon in."""
        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return None
        chart = self._chart(battle)
        best = 0.0
        for t in opponent.types:
            if t is not None:
                best = max(best, t.damage_multiplier(*mon.types, type_chart=chart))
        return best

    @staticmethod
    def _fixed_damage(move):
        """Guaranteed HP damage for fixed-damage moves (Seismic Toss, Night Shade,
        Dragon Rage...), which report base_power 0. 'level' -> 100 (we play at L100)."""
        d = getattr(move, "damage", None)
        if not d:
            return None
        if d == "level":
            return 100.0
        try:
            return float(d)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _team_mon(mon) -> TeamMon:
        return TeamMon(
            species=mon.species,
            hp_fraction=mon.current_hp_fraction,
            fainted=bool(mon.fainted),
            status=mon.status.name if mon.status else None,
            active=bool(getattr(mon, "active", False)),
        )
