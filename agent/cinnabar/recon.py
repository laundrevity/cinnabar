"""Browser ground truth: reconstruct the live Showdown battle as an engine battle, so the
k-gated search agent can play humans.

Decision-time search needs a `ce.Battle` to clone — which only exists on the engine path. This
module rebuilds one PER DECISION from poke-env's view of the real battle (the second and last
module allowed to touch poke-env, alongside showdown.py):

  * our side: full team + movesets (we supplied the team), exact HP/status/boosts,
  * their side: REVEALED mons with REVEALED moves, padded from the species' metagame movepool;
    unrevealed slots get a placeholder — partial information stays partial,
  * actives, screens, confusion, Leech Seed, sleep/toxic counters injected via the engine's
    state-injection API (set_mon_state / set_active_slot).

Known approximations (documented; search is robust to leaf noise): PP is reset to full,
Substitute/Disable/partial-trap are not reconstructed, sleep-turns-remaining and confusion turns
are estimates, and a boost-then-status history reorders the Gen 1 stat-drop bug. If anything fails,
`SearchPolicyPlayer` falls back to the raw policy — a live game never hangs on reconstruction.
"""

from __future__ import annotations

from .engine_cpp import StaticData, _to_id, build_state
from .search import search_action_values
from .showdown import PolicyPlayer

import cinnabar_engine as ce  # noqa: E402  (importable after engine_cpp set the path)

FILLER_MOVE = "Body Slam"          # engine-modelled, plausible on anything
PLACEHOLDER = ("Tauros", ["Body Slam", "Hyper Beam", "Earthquake", "Blizzard"])  # unrevealed slot


class ReconError(RuntimeError):
    pass


def _display_move(static: StaticData, move_id: str) -> str:
    return (static._moves.get(_to_id(move_id), {}) or {}).get("name", move_id)


def _display_species(static: StaticData, species: str) -> str:
    return (static._dex.get(_to_id(species), {}) or {}).get("name", species)


def _moveset(static, mon, pad_from_pool: bool) -> list[str]:
    """Engine move names for a poke-env mon: its (known or revealed) moves, padded to 4."""
    names = [_display_move(static, mid) for mid in (mon.moves or {})][:4]
    if pad_from_pool and len(names) < 4:
        try:
            from . import movesets
            pool = movesets.SPECIES_MOVEPOOLS.get(_display_species(static, mon.species), [])
        except Exception:
            pool = []
        for mv in pool:
            if len(names) >= 4:
                break
            if mv not in names:
                names.append(mv)
    if not names:
        names = [FILLER_MOVE]
    return names


def _status_code(mon) -> tuple[str, int, int]:
    """(engine status code, sleep_turns_remaining_estimate, tox_stage) for a poke-env mon."""
    st = mon.status.name if mon.status else ""
    counter = int(getattr(mon, "status_counter", 0) or 0)
    if st == "SLP":
        return "SLP", max(1, 4 - counter), 0   # remaining turns unknown to the client: estimate
    if st == "TOX":
        return "TOX", 0, max(1, counter)
    return st, 0, 0


def _effects(mon) -> dict:
    names = {e.name for e in (getattr(mon, "effects", None) or {})}
    return {
        "reflect": "REFLECT" in names,
        "light_screen": "LIGHT_SCREEN" in names,
        "leech_seeded": "LEECH_SEED" in names,
        "confuse_turns": 2 if "CONFUSION" in names else 0,  # remaining turns: estimate
        "must_recharge": "MUST_RECHARGE" in names or "MUSTRECHARGE" in names,
    }


def _boosts(mon) -> tuple[int, int, int, int]:
    b = getattr(mon, "boosts", None) or {}
    return (b.get("atk", 0), b.get("def", 0), b.get("spa", 0), b.get("spe", 0))


def _inject(battle, player: int, mons, actives_extra: bool) -> None:
    """Push each poke-env mon's condition into the recon battle at its slot."""
    for slot, mon in enumerate(mons):
        st, slp, tox = _status_code(mon)
        kw = {}
        if getattr(mon, "active", False) and actives_extra:
            ba, bd, bs, bp = _boosts(mon)
            kw.update(boost_atk=ba, boost_def=bd, boost_spc=bs, boost_spe=bp, **_effects(mon))
        battle.set_mon_state(player, slot, 0.0 if mon.fainted else float(mon.current_hp_fraction),
                             status=("" if mon.fainted else st), sleep_turns=slp,
                             tox_stage=tox, **kw)
        if getattr(mon, "active", False):
            battle.set_active_slot(player, slot)


def reconstruct(pe_battle, static: StaticData, clauses: bool = True, seed: int = 1):
    """poke-env Battle -> (ce.Battle at the current position, spec1 (us), spec2 (them)).
    Raises ReconError on anything unmappable; callers fall back to the raw policy."""
    ours = list(pe_battle.team.values())
    theirs = list(pe_battle.opponent_team.values())
    if not ours or pe_battle.active_pokemon is None:
        raise ReconError("no team view yet")

    spec1 = [(_display_species(static, m.species), _moveset(static, m, pad_from_pool=False))
             for m in ours]
    spec2 = [(_display_species(static, m.species), _moveset(static, m, pad_from_pool=True))
             for m in theirs]
    while len(spec2) < 6:  # unrevealed opponents: placeholder (partial info stays partial)
        spec2.append((PLACEHOLDER[0], list(PLACEHOLDER[1])))

    try:
        battle = ce.make_battle([(s, list(m)) for s, m in spec1],
                                [(s, list(m)) for s, m in spec2], seed)
    except Exception as e:  # unknown species/move for the engine
        raise ReconError(f"make_battle: {e}") from e
    if clauses:
        battle.set_clauses(True)

    _inject(battle, 0, ours, actives_extra=True)
    _inject(battle, 1, theirs, actives_extra=True)
    if not any(getattr(m, "active", False) for m in ours):
        raise ReconError("no active on our side")
    return battle, spec1, spec2


class SearchPolicyPlayer(PolicyPlayer):
    """PolicyPlayer that decides by k-gated decision-time search on a per-turn reconstruction of
    the live battle — the browser agent finally gets the search edge. Any reconstruction or
    mapping failure falls back to the wrapped raw policy (the game never hangs)."""

    def __init__(self, policy, net, opp_model, *args, rollouts: int = 3, top_k: int = 3,
                 clauses: bool = True, device: str = "cpu", **kwargs) -> None:
        super().__init__(policy, *args, **kwargs)
        self._net = net
        self._opp_model = opp_model
        self._rollouts = rollouts
        self._top_k = top_k
        self._clauses = clauses
        self._device = device
        self._recon_seed = 1000
        self.search_moves = 0
        self.fallback_moves = 0

    def choose_move(self, battle):
        try:
            order = self._search_order(battle)
            if order is not None:
                self.search_moves += 1
                return order
        except Exception as e:  # noqa: BLE001 — any recon failure means: play the raw policy
            print(f"  [recon fallback @ turn {getattr(battle, 'turn', '?')}: {e}]")
        self.fallback_moves += 1
        return super().choose_move(battle)

    def _search_order(self, battle):
        recon, spec1, spec2 = reconstruct(battle, self._static, clauses=self._clauses,
                                          seed=self._recon_seed)
        self._recon_seed += 1
        state = build_state(recon, 0, spec1, static=self._static, tag="brs",
                            reveal=None, opp_team=spec2)
        if not state.available_actions:
            return None
        cands, values = search_action_values(
            recon, 0, self._net, self._opp_model, self._static, spec1, spec2,
            reveal=None, device=self._device, rollouts=self._rollouts,
            state=state, top_k=self._top_k)
        # Best-first, take the first candidate that maps onto a real Showdown order. (The recon
        # battle can offer moves the real battle forbids — Disabled, out of PP — and vice versa.)
        order = sorted(zip(cands, values), key=lambda cv: -cv[1])
        active_slot = next(i for i, m in enumerate(battle.team.values())
                           if getattr(m, "active", False))
        my_moves = spec1[active_slot][1]
        for ci, _v in order:
            choice = recon.choices(0)[ci]
            if choice.kind == ce.ChoiceKind.Move:
                if choice.index < 0:
                    continue  # Struggle/recharge: let poke-env's default handle the forced case
                want = _to_id(my_moves[choice.index])
                for mv in battle.available_moves:
                    if mv.id == want:
                        return self.create_order(mv)
            elif choice.kind == ce.ChoiceKind.Switch:
                want = _to_id(spec1[choice.index][0])
                for mon in battle.available_switches:
                    if _to_id(mon.species) == want:
                        return self.create_order(mon)
        return None  # nothing mappable: fall back
