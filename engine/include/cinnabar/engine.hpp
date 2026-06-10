// Cinnabar Gen 1 battle engine (C++). See engine/README.md for scope + fidelity strategy.
//
// Covers: 6-Pokémon teams + switching (with forced switch on faint), status
// (sleep/freeze/paralysis/burn/poison) and the moves that cause them, healing
// (Recover/Rest), Reflect, Explosion/Self-Destruct, fixed-damage moves, stat stages,
// PP/Struggle, Hyper Beam recharge, recoil, confusion (Confuse Ray), Substitute, Counter,
// partial-trapping (Wrap/Fire Spin/Clamp/Bind, incl. the Residual fieldEvent shuffle RNG),
// and the Gen 1 damage/stat/type formulas — all validated bit-for-bit against Showdown (Gen 5 LCG RNG).
//
// Not yet modelled (refine via differential testing vs Showdown): freeze thaw,
// multi-turn locking moves (Thrash/Petal Dance), charge/semi-invulnerable moves (Dig/Fly/Sky Attack).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cinnabar {

enum class Type {
    Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost,
    Fire, Water, Grass, Electric, Psychic, Ice, Dragon, None,
};

enum class Status { None, Sleep, Poison, Burn, Freeze, Paralysis };
enum class Category { Physical, Special, Status };
enum class Result { Ongoing, P1Win, P2Win, Tie };

// Primary or secondary move effect.
enum class Effect { None, Paralyze, Sleep, Freeze, Burn, Poison, Heal, Rest, Reflect, SelfDestruct,
                    Substitute, Confuse, Counter, Trap, SuperFang, Psywave, LightScreen, LeechSeed,
                    Toxic, Disable };

double type_effectiveness(Type attacking, Type defending);  // Gen 1 (provisional)

// Effectiveness is applied per defending type, in order (def_t1 then def_t2), flooring
// after each step like Showdown (×20/10 super, ×5/10 resisted) — not as one combined factor.
int gen1_damage(int level, int power, int attack, int defense, bool stab,
                Type move_type, Type def_t1, Type def_t2, bool crit, int random);

struct Species {
    std::string name;
    Type t1, t2;
    int hp, atk, def, spc, spe;  // base stats
};

struct MoveData {
    std::string name;
    Type type;
    Category category;  // informational; Gen 1 uses type for phys/spec
    int power;          // 0 for status / fixed-damage
    int accuracy;       // out of 100
    int fixed = 0;      // fixed HP damage (Seismic Toss = 100), else 0
    Effect effect = Effect::None;
    int effect_chance = 0;  // % chance of the effect (status moves use 100)
    // Stat-stage change. boost_stat: -1 none, 0=atk 1=def 2=spc 3=spe (4=acc 5=eva, unmodelled).
    int boost_stat = -1;
    int boost_stages = 0;            // signed number of stages
    bool boost_target_foe = false;   // false: user (Amnesia/Swords Dance); true: foe (Psychic 2ndary)
    int boost_chance = 0;            // 0 = always (status move); else % chance (a 2ndary)
    bool high_crit = false;          // high crit-ratio moves (Slash, Razor Leaf, Crabhammer, ...)
    int pp = 0;                      // base PP (0 = untracked, e.g. hand-built test moves)
    bool recharge = false;           // Hyper Beam: user must spend next turn recharging unless it KO'd
    int recoil_num = 0, recoil_den = 0;  // recoil to the user = floor(damage * num/den), min 1
    bool ignore_immunity = false;        // skip the type-immunity check (Confuse Ray, Glare, ...)
    int priority = 0;                    // turn-order bracket; Counter is -5 (moves last)
    bool skip_lastdamage = false;        // does NOT reset the battle's last_damage (Counter, status)
    int drain_num = 0, drain_den = 0;    // drain heal to the user = floor(damage * num/den), min 1 (Mega Drain)
    bool needs_sleep_target = false;     // Dream Eater: fails (before accuracy) unless the target is asleep
    int multihit_min = 0, multihit_max = 0;  // multi-hit count range (0 = single hit); (2,5) = the
                                             // gen1 sample distribution, min==max = a fixed count (Double Kick)
};

struct Pokemon {
    const Species* species = nullptr;
    int level = 100;
    int max_hp = 0, hp = 0;
    int atk = 0, def = 0, spc = 0, spe = 0;          // stored (base) stats — immutable in battle
    int m_atk = 0, m_def = 0, m_spc = 0, m_spe = 0;  // modifiedStats (boosts + burn/paralysis)
    int boost_atk = 0, boost_def = 0, boost_spc = 0, boost_spe = 0;  // stages, -6..+6
    Status status = Status::None;
    int sleep_turns = 0;
    bool status_by_foe = false;  // this Sleep/Freeze was inflicted by a foe's move (not Rest) — for
                                 // Sleep/Freeze Clause: only one foe-inflicted sleep|freeze per side
    bool toxic = false;   // badly poisoned (Toxic): reports 'tox' and deals escalating residual damage
    int tox_stage = 0;    // toxic counter; residual = tox_stage * floor(maxhp/16), +1 per turn; resets on switch
    bool reflect = false;        // volatile: Reflect screen (doubles Def), cleared on switch out
    bool light_screen = false;   // volatile: Light Screen (doubles Spc def), cleared on switch out
    bool leech_seeded = false;   // volatile: Leech Seed drains 1/16 max HP after this mon's move; cleared on switch
    int disable_slot = -1;       // Disable: the move slot that's disabled (-1 = none); cleared on switch
    int disable_turns = 0;       // Disable: turns the slot stays disabled (1-8); ticks down each of this mon's moves
    bool must_recharge = false;  // volatile: owes a Hyper Beam recharge turn; cleared on switch
    bool has_substitute = false; // volatile: a Substitute is up, absorbing damage; cleared on switch
    int sub_hp = 0;              // remaining Substitute HP (floor(maxhp/4)+1 when created)
    int confuse_turns = 0;      // volatile: remaining confusion turns (0 = not confused); cleared on switch
    int wrap_turns = 0;         // wrapper: partial-trap lock duration remaining (re-uses wrap_idx while >0)
    int wrap_total = 0;         // the sampled lock duration (fakepartiallytrapped is skipped when this is 5)
    int wrap_idx = -1;          // the move slot the wrapper is locked into (Wrap/Bind/Fire Spin/Clamp)
    int wrap_damage = 0;        // first-turn damage, re-dealt each turn (Gen 1 partial-trap is fixed)
    int partial_trapped = 0;    // victim: partiallytrapped duration remaining (loses its turn while >0)
    int fake_trap_turns = 0;    // fakepartiallytrapped: 2-turn cosmetic volatile (its onDisableMove handler
                                // joins Showdown's fieldEvent speed-sort, consuming shuffle RNG at boundaries)
    std::vector<const MoveData*> moves;
    std::vector<int> pp;  // current PP per move slot (parallel to moves); -1 = untracked/unlimited

    bool fainted() const { return hp <= 0; }
    double hp_fraction() const { return max_hp ? static_cast<double>(hp) / max_hp : 0.0; }
};

Pokemon make_pokemon(const Species* s, std::vector<const MoveData*> moves, int level = 100);

// Look up a Gen 1 species by name from the generated data (gen1_data.hpp).
const Species& species(const std::string& name);

// Look up a move by name from the engine's move table (hand-coded for now).
const MoveData& move(const std::string& name);

// Bit-identical to Showdown's Gen 5 LCG PRNG. Seed Showdown battles with a numeric/gen5
// seed (not the default 'sodium' ChaCha20) so its RNG stream matches this one exactly.
struct RNG {
    uint64_t state;
    explicit RNG(uint64_t seed) : state(seed) {}
    uint32_t next();               // advance the LCG, return the upper 32 bits
    int random(int n);             // [0, n)        == Showdown PRNG.random(n)
    int random(int from, int to);  // [from, to)    == Showdown PRNG.random(from, to)
    int range(int lo, int hi);     // inclusive [lo, hi] == random(lo, hi + 1)
    bool chance(int num, int den); // == Showdown PRNG.randomChance(num, den)
};

struct Side {
    std::vector<Pokemon> team;  // up to 6
    int active = 0;
    bool must_switch = false;  // active fainted, a replacement is required
    bool clauses = false;      // OU Sleep + Freeze Clause enabled (training only; off for the
                               // bit-for-bit fidelity harness, which validates vs clause-free customgame)
    // For Counter (Gen 1): the last move this side executed and last move it selected. Both are
    // per-side and persist across switches (Counter can reflect damage from a since-switched mon).
    const MoveData* last_move = nullptr;
    const MoveData* last_selected = nullptr;

    Pokemon& mon() { return team[active]; }
    const Pokemon& mon() const { return team[active]; }
    bool has_alive_bench() const;
    bool all_fainted() const;
};

enum class ChoiceKind { Move, Switch, Pass };
struct Choice {
    ChoiceKind kind = ChoiceKind::Pass;
    int index = 0;  // Move: move slot; Switch: team index
};
inline Choice move_choice(int i) { return {ChoiceKind::Move, i}; }
inline Choice switch_choice(int i) { return {ChoiceKind::Switch, i}; }
inline Choice pass_choice() { return {ChoiceKind::Pass, 0}; }

struct Battle {
    Side p1, p2;
    RNG rng;
    int turn = 0;
    int last_damage = 0;  // most recent damage dealt in the battle (what Counter doubles)

    Battle(Side a, Side b, uint64_t seed) : p1(std::move(a)), p2(std::move(b)), rng(seed) {}

    // Enable/disable OU Sleep + Freeze Clause on both sides. Off by default (keeps the fidelity
    // harness validating against clause-free customgame); training turns it on so the agent learns
    // that a second foe-inflicted sleep/freeze fails — instead of spamming sleep as a free win.
    void set_clauses(bool on) { p1.clauses = p2.clauses = on; }

    // Reseed the internal PRNG (used only on *clones* during decision-time search, so rollouts draw
    // fresh dice instead of replaying the live game's RNG — the real battle's rng is never touched).
    void reseed(uint64_t s) { rng = RNG(s); }

    std::vector<Choice> choices(int player) const;  // 0 = p1, 1 = p2
    Result step(const Choice& c1, const Choice& c2);
    Result result() const;
};

// Build a battle from team specs: each entry is (species name, [move names]).
using TeamSpec = std::vector<std::pair<std::string, std::vector<std::string>>>;
Battle make_battle(const TeamSpec& team1, const TeamSpec& team2, uint64_t seed);

// --- State injection (browser ground-truth reconstruction) ------------------------------------
// Additive API: unused by training, search-on-engine, and the bit-for-bit fidelity harness.
// Sets one mon's mid-battle condition and recomputes its modified stats in the canonical order:
// stages first (boostBy), then the burn/paralysis drop (modifyStat). A mon that was boosted
// AFTER being statused would have the drop discarded (the real Gen 1 boostBy bug) — that history
// is unknowable from outside, so this is a documented approximation. PP, Substitute, Disable and
// partial-trap state are left at defaults (reconstruction gaps; see agent/cinnabar/recon.py).
void set_mon_state(Battle& b, int player, int slot, double hp_fraction, Status status,
                   int sleep_turns, bool status_by_foe, int stage_atk, int stage_def,
                   int stage_spc, int stage_spe, bool reflect, bool light_screen,
                   bool must_recharge, bool toxic, int tox_stage, int confuse_turns,
                   bool leech_seeded);
// Choose the active slot; sets must_switch when that mon is fainted and a bench remains.
void set_active_slot(Battle& b, int player, int slot);

}  // namespace cinnabar
