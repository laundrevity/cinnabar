// Cinnabar Gen 1 battle engine (C++). See engine/README.md for scope + fidelity strategy.
//
// Covers: 6-Pokémon teams + switching (with forced switch on faint), status
// (sleep/freeze/paralysis/burn/poison) and the moves that cause them, healing
// (Recover/Rest), Reflect, Explosion/Self-Destruct, fixed-damage moves, stat stages,
// PP/Struggle, Hyper Beam recharge, and the Gen 1 damage/stat/type formulas — all
// validated bit-for-bit against Showdown (Gen 5 LCG RNG).
//
// Not yet modelled (refine via differential testing vs Showdown): freeze thaw, confusion,
// Substitute, Counter, partial-trapping (Wrap/Fire Spin), multi-turn moves (Thrash/Dig/Fly).
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
                    Substitute, Confuse, Counter, Trap };

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
    bool reflect = false;        // volatile: Reflect screen, cleared on switch out
    bool must_recharge = false;  // volatile: owes a Hyper Beam recharge turn; cleared on switch
    bool has_substitute = false; // volatile: a Substitute is up, absorbing damage; cleared on switch
    int sub_hp = 0;              // remaining Substitute HP (floor(maxhp/4)+1 when created)
    int confuse_turns = 0;      // volatile: remaining confusion turns (0 = not confused); cleared on switch
    int wrap_turns = 0;         // wrapper: partial-trap lock duration remaining (re-uses wrap_idx while >0)
    int wrap_idx = -1;          // the move slot the wrapper is locked into (Wrap/Bind/Fire Spin/Clamp)
    int wrap_damage = 0;        // first-turn damage, re-dealt each turn (Gen 1 partial-trap is fixed)
    int partial_trapped = 0;    // victim: partiallytrapped duration remaining (loses its turn while >0)
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

    std::vector<Choice> choices(int player) const;  // 0 = p1, 1 = p2
    Result step(const Choice& c1, const Choice& c2);
    Result result() const;
};

// Build a battle from team specs: each entry is (species name, [move names]).
using TeamSpec = std::vector<std::pair<std::string, std::vector<std::string>>>;
Battle make_battle(const TeamSpec& team1, const TeamSpec& team2, uint64_t seed);

}  // namespace cinnabar
