// Cinnabar Gen 1 battle engine (C++). See engine/README.md for scope + fidelity strategy.
//
// Covers: 6-Pokémon teams + switching (with forced switch on faint), status
// (sleep/freeze/paralysis/burn/poison) and the moves that cause them, healing
// (Recover/Rest), Reflect, Explosion/Self-Destruct, fixed-damage moves, and the
// Gen 1 damage formula / stat formula / type chart.
//
// Known simplifications (to refine via differential testing vs Showdown): the
// 1/256 miss, exact crit rate, freeze thaw, PP/Struggle, stat stages, Hyper Beam
// recharge, and Showdown-bit-compatible RNG are not yet modelled.
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
enum class Effect { None, Paralyze, Sleep, Freeze, Burn, Poison, Heal, Rest, Reflect, SelfDestruct };

double type_effectiveness(Type attacking, Type defending);  // Gen 1 (provisional)

int gen1_damage(int level, int power, int attack, int defense,
                bool stab, double type_mult, bool crit, int random);

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
};

struct Pokemon {
    const Species* species = nullptr;
    int level = 100;
    int max_hp = 0, hp = 0;
    int atk = 0, def = 0, spc = 0, spe = 0;
    Status status = Status::None;
    int sleep_turns = 0;
    bool reflect = false;  // volatile: Reflect screen, cleared on switch out
    std::vector<const MoveData*> moves;

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

    Battle(Side a, Side b, uint64_t seed) : p1(std::move(a)), p2(std::move(b)), rng(seed) {}

    std::vector<Choice> choices(int player) const;  // 0 = p1, 1 = p2
    Result step(const Choice& c1, const Choice& c2);
    Result result() const;
};

// Build a battle from team specs: each entry is (species name, [move names]).
using TeamSpec = std::vector<std::pair<std::string, std::vector<std::string>>>;
Battle make_battle(const TeamSpec& team1, const TeamSpec& team2, uint64_t seed);

}  // namespace cinnabar
