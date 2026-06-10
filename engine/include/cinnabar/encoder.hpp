// C++ observation encoder — the throughput-critical mirror of the Python featurization
// (agent/cinnabar/engine_cpp.py build_state + agent/cinnabar/encoding.py featurize).
//
// Profiling showed the engine itself runs ~250k turns/sec but the Python per-turn
// state-building collapsed training to a few hundred battles/sec — ~80% of rollout time
// was build_state and friends. This module computes the exact same GLOBAL_DIM/ACTION_DIM
// feature vectors directly from the Battle, in C++.
//
// Fidelity strategy (same spirit as the engine-vs-Showdown harness): the static
// move/species metadata and the type chart are NOT re-derived here — Python registers
// them once at startup from the same poke-env tables build_state uses, so this encoder
// is pure arithmetic on identical inputs. A pytest (agent/tests/test_encoder_parity.py)
// asserts bit-identical output against the Python encoder across whole battles.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "cinnabar/engine.hpp"

namespace cinnabar::enc {

constexpr int kTypes = 15;        // encoding.TYPE_ORDER (== the engine's Type enum order)
constexpr int kTeam = 6;
constexpr int kStatusOneHot = 6;  // encoding.STATUS_ORDER: NONE BRN FRZ PAR PSN SLP
constexpr int kOppMonDim = 4 + kTypes;
constexpr int kOppMoveDim = 7;
constexpr int kGlobalDim = 2 + 2 * kStatusOneHot + 2 + 4 + 1 + 12 + 12
                           + kTeam * kOppMonDim + kOppMoveDim + 2;  // == encoding.GLOBAL_DIM
constexpr int kActionDim = 7 + 3 + 1 + 11 + 1;                      // == encoding.ACTION_DIM

// Per-move metadata as Python's StaticData.move_meta computes it from poke-env (the
// defaults below must equal move_meta's unknown-move fallbacks, e.g. the synthetic
// "Recharge" choice). h_* are the heuristic policies' hand-picked move sets.
struct MoveMeta {
    double base_power = 0.0;
    double accuracy = 1.0;
    double fixed = 0.0;       // guaranteed HP (Seismic Toss = 100); 0 = none
    double effect_chance = 0.0;
    int type_idx = 0;         // index into TYPE_ORDER; NORMAL default
    int effect_status = -1;   // index into {SLP, PAR, FRZ, BRN, PSN}; -1 = none
    bool is_status = true;    // category == STATUS
    bool heals = false, boosts_self = false, lowers_foe = false;
    bool recharge = false, self_destruct = false;
    bool h_sleep = false, h_para = false, h_heal = false, h_hyperbeam = false;
};

struct SpeciesMeta {
    std::array<int8_t, 2> types{-1, -1};  // TYPE_ORDER indices, dex order; -1 = no slot
    int speed = -1;                       // base Speed; -1 = unknown
};

// Per-battle partial-information memory, the C++ twin of engine_cpp.Reveal: which
// opponent team slots have been seen, and which of each slot's move slots. Trivially
// copyable, so decision-time search can snapshot it per leaf for free.
struct Observer {
    uint8_t mons = 0;                // revealed opponent team slots (bitmask)
    std::array<uint8_t, kTeam> moves{};  // per opponent slot: revealed move slots (bitmask)
};

// Registration (called once from Python with poke-env-derived data).
void register_type_chart(const double chart_def_atk[kTypes][kTypes]);  // [defender][attacker]
void register_move(const std::string& id, const MoveMeta& m);          // id = lowercase alnum
void register_species(const std::string& id, const SpeciesMeta& s);
bool encoder_ready();

// Encode `player`'s view of the battle. Writes kGlobalDim floats to `g` and one
// kActionDim row per legal action to `a` (row-major; caller sizes it from
// choices(player).size()); returns the action count. Mutates `obs` exactly like
// build_state mutates Reveal (the opponent's current active becomes revealed).
// obs == nullptr is the full-information view (build_state's reveal=None), which also
// leaves the revealed-move threat profile empty, matching the Python adapter.
// my_mat/opp_mat (optional): HP-fraction sums — ours over the full team, theirs over
// revealed mons only — the dense-reward material bookkeeping from train_engine.
int encode(const Battle& b, int player, Observer* obs, float* g, float* a,
           double* my_mat = nullptr, double* opp_mat = nullptr);

// One call per turn: record each side's selected move into the *other* side's observer
// (mirroring engine_cpp.reveal_move: moves only, Struggle/Recharge excluded), then step.
// i1/i2 index into choices(0)/choices(1).
Result step_pair(Battle& b, int i1, int i2, Observer* obs1, Observer* obs2);

// C++ twins of the heuristic pilots in agent/cinnabar/policy.py (decision-for-decision
// parity, tested). kind: 0 = MaxDamagePolicy, 1 = SmartHeuristicPolicy, 2 = StallerPolicy.
// Mutates obs like encode does (the policies act on the same just-revealed view).
int select_heuristic(const Battle& b, int player, Observer* obs, int kind);

// The whole decision-time-search leaf loop in one call (search.py's hot path): for every
// (my candidate x opponent reply x rollout), clone the battle, reseed with the given dice,
// step the turn, then resolve the leaf:
//   mode 0 — encode the post-step state (value-head leaf; obs snapshot per leaf);
//   mode 1 — heuristic playout to terminal (rollout leaf, capped at `cap` turns);
//   mode 2 — `depth` heuristic turns, then terminal outcome or full-info encode (bootstrap).
// Leaf order is my_cands-major, then opp_cands, then rollouts — matching the Python loops'
// RNG consumption. seeds.size() must equal my_cands*opp_cands*rollouts. For each leaf l:
// direct[l] = the terminal/playout value, or NaN meaning "evaluate rows[l] with the net".
void search_leaves(const Battle& b, int player, const std::vector<int>& my_cands,
                   const std::vector<int>& opp_cands, int rollouts,
                   const std::vector<uint64_t>& seeds, const Observer* obs,
                   int mode, int rkind, int depth, int cap,
                   float* rows, double* direct);

}  // namespace cinnabar::enc
