#include "cinnabar/engine.hpp"

#include "cinnabar/gen1_data.hpp"  // generated from Showdown: GEN1_TYPE_CHART, GEN1_SPECIES

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace cinnabar {

double type_effectiveness(Type attacking, Type defending) {
    if (attacking == Type::None || defending == Type::None) return 1.0;
    return GEN1_TYPE_CHART[static_cast<int>(attacking)][static_cast<int>(defending)] / 100.0;
}

const Species& species(const std::string& name) {
    static const std::unordered_map<std::string, Species> table = [] {
        std::unordered_map<std::string, Species> m;
        for (const auto& e : GEN1_SPECIES) {
            Species s;
            s.name = e.name;
            s.t1 = e.t1;
            s.t2 = e.t2;
            s.hp = e.hp; s.atk = e.atk; s.def = e.def; s.spc = e.spc; s.spe = e.spe;
            m.emplace(e.name, s);
        }
        return m;
    }();
    auto it = table.find(name);
    if (it == table.end()) throw std::out_of_range("unknown species: " + name);
    return it->second;
}

const MoveData& move(const std::string& name) {
    static const std::unordered_map<std::string, MoveData> table = [] {
        std::unordered_map<std::string, MoveData> m;
        auto add = [&](MoveData md) { m.emplace(md.name, md); };
        // Hand-coded for now (the moves our teams use); a codegen pass will source
        // these from Showdown like gen1_data.hpp does for species.
        add({"Psychic", Type::Psychic, Category::Special, 90, 100});
        add({"Thunder Wave", Type::Electric, Category::Status, 0, 100, 0, Effect::Paralyze, 100});
        add({"Recover", Type::Normal, Category::Status, 0, 100, 0, Effect::Heal, 100});
        add({"Soft-Boiled", Type::Normal, Category::Status, 0, 100, 0, Effect::Heal, 100});
        add({"Seismic Toss", Type::Fighting, Category::Status, 0, 100, 100});
        add({"Ice Beam", Type::Ice, Category::Special, 95, 100, 0, Effect::Freeze, 10});
        add({"Thunderbolt", Type::Electric, Category::Special, 95, 100, 0, Effect::Paralyze, 10});
        add({"Sleep Powder", Type::Grass, Category::Status, 0, 75, 0, Effect::Sleep, 100});
        add({"Explosion", Type::Normal, Category::Physical, 170, 100, 0, Effect::SelfDestruct, 0});
        add({"Body Slam", Type::Normal, Category::Physical, 85, 100, 0, Effect::Paralyze, 30});
        add({"Earthquake", Type::Ground, Category::Physical, 100, 100});
        add({"Blizzard", Type::Ice, Category::Special, 120, 90, 0, Effect::Freeze, 10});
        add({"Hyper Beam", Type::Normal, Category::Physical, 150, 90});
        add({"Rest", Type::Psychic, Category::Status, 0, 100, 0, Effect::Rest, 100});
        return m;
    }();
    auto it = table.find(name);
    if (it == table.end()) throw std::out_of_range("unknown move: " + name);
    return it->second;
}

int gen1_damage(int level, int power, int attack, int defense, bool stab, double type_mult,
                bool crit, int random) {
    if (power <= 0 || type_mult == 0.0) return 0;
    int L = crit ? 2 * level : level;
    long dmg = ((2L * L) / 5 + 2);
    dmg = dmg * power * attack / std::max(1, defense);
    dmg = dmg / 50 + 2;
    if (stab) dmg = dmg * 3 / 2;
    dmg = static_cast<long>(dmg * type_mult);
    dmg = dmg * random / 255;
    if (type_mult > 0.0 && dmg < 1) dmg = 1;
    return static_cast<int>(dmg);
}

Pokemon make_pokemon(const Species* s, std::vector<const MoveData*> moves, int level) {
    Pokemon p;
    p.species = s;
    p.level = level;
    auto stat = [&](int base) { return (2 * (base + 15) + 63) * level / 100 + 5; };
    p.max_hp = (2 * (s->hp + 15) + 63) * level / 100 + level + 10;
    p.hp = p.max_hp;
    p.atk = stat(s->atk);
    p.def = stat(s->def);
    p.spc = stat(s->spc);
    p.spe = stat(s->spe);
    p.moves = std::move(moves);
    return p;
}

uint32_t RNG::next() {
    // Showdown Gen5RNG: x = x * a + c (mod 2^64); output is the upper 32 bits.
    state = state * 0x5D588B656C078965ULL + 0x00269EC3ULL;
    return static_cast<uint32_t>(state >> 32);
}
int RNG::random(int n) {
    return static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(n)) >> 32);
}
int RNG::random(int from, int to) {
    return static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(to - from)) >> 32) + from;
}
int RNG::range(int lo, int hi) { return random(lo, hi + 1); }
bool RNG::chance(int num, int den) { return random(den) < num; }

bool Side::has_alive_bench() const {
    for (int i = 0; i < static_cast<int>(team.size()); ++i)
        if (i != active && !team[i].fainted()) return true;
    return false;
}
bool Side::all_fainted() const {
    for (const auto& m : team)
        if (!m.fainted()) return false;
    return true;
}

namespace {
int effective_speed(const Pokemon& p) {
    return p.status == Status::Paralysis ? p.spe / 4 : p.spe;
}

// Gen 1: a move is physical/special purely by its type (no per-move split).
bool gen1_is_physical(Type t) {
    switch (t) {
        case Type::Normal: case Type::Fighting: case Type::Flying: case Type::Poison:
        case Type::Ground: case Type::Rock: case Type::Bug: case Type::Ghost:
            return true;
        default:
            return false;
    }
}

double combined_effectiveness(Type atk, const Species* def) {
    double e = type_effectiveness(atk, def->t1);
    if (def->t2 != Type::None) e *= type_effectiveness(atk, def->t2);
    return e;
}

bool can_act(Pokemon& p, RNG& rng) {
    switch (p.status) {
        case Status::Freeze:
            return false;  // Gen 1: frozen solid (thaw not modelled yet)
        case Status::Sleep:
            if (p.sleep_turns > 0) --p.sleep_turns;
            if (p.sleep_turns <= 0) p.status = Status::None;
            return false;  // Gen 1: a turn is always lost to sleep, including the wake turn
        case Status::Paralysis:
            return !rng.chance(1, 4);  // 25% fully paralyzed
        default:
            return true;
    }
}

void apply_effect(const MoveData* mv, Pokemon& user, Pokemon& tgt, RNG& rng) {
    if (mv->effect == Effect::None) return;
    auto roll = [&]() { return mv->effect_chance >= 100 || rng.chance(mv->effect_chance, 100); };
    switch (mv->effect) {
        case Effect::Heal:
            user.hp = std::min(user.max_hp, user.hp + user.max_hp / 2);
            return;
        case Effect::Rest:
            user.status = Status::Sleep;
            user.sleep_turns = 2;
            user.hp = user.max_hp;
            return;
        case Effect::Reflect:
            user.reflect = true;
            return;
        case Effect::SelfDestruct:
            return;
        default:
            break;
    }
    if (tgt.fainted() || tgt.status != Status::None) return;
    if (mv->power == 0 && mv->fixed == 0 && combined_effectiveness(mv->type, tgt.species) == 0.0)
        return;  // status move blocked by type immunity (e.g. Thunder Wave vs Ground)
    if (!roll()) return;
    switch (mv->effect) {
        case Effect::Paralyze: tgt.status = Status::Paralysis; break;
        case Effect::Sleep:    tgt.status = Status::Sleep; tgt.sleep_turns = rng.range(1, 7); break;
        case Effect::Freeze:   tgt.status = Status::Freeze; break;
        case Effect::Burn:     tgt.status = Status::Burn; break;
        case Effect::Poison:   tgt.status = Status::Poison; break;
        default: break;
    }
}

void use_move(Side& as, Side& ds, int moveidx, RNG& rng) {
    Pokemon& a = as.mon();
    Pokemon& d = ds.mon();
    if (moveidx < 0 || moveidx >= static_cast<int>(a.moves.size())) return;
    const MoveData* mv = a.moves[moveidx];
    if (!mv) return;

    // Accuracy — Showdown gen1 rolls randomChance(clamp(floor(acc*255/100),1,255), 256) for
    // every non-self-targeting move, *including* 100%-accuracy ones (the 1/256 miss). Moves
    // that target the user (Recover/Rest/Reflect) skip the roll.
    bool self_targeting = mv->effect == Effect::Heal || mv->effect == Effect::Rest ||
                          mv->effect == Effect::Reflect;
    if (!self_targeting && mv->accuracy > 0) {
        int acc = std::clamp(mv->accuracy * 255 / 100, 1, 255);
        if (!rng.chance(acc, 256)) return;  // miss (includes the gen1 1/256)
    }

    if (mv->fixed > 0) {
        if (combined_effectiveness(mv->type, d.species) == 0.0) return;
        d.hp -= mv->fixed;
    } else if (mv->power > 0 && !d.fainted()) {
        double mult = combined_effectiveness(mv->type, d.species);
        if (mult == 0.0) return;  // immune: no damage, no secondary
        bool stab = (mv->type == a.species->t1 || mv->type == a.species->t2);
        // Crit — Showdown gen1 derives the chance from BASE species Speed, not the live stat:
        // critChance = clamp(floor(baseSpe/2)*2, 1, 255), then /2 for a normal crit ratio.
        int crit_chance = std::clamp((a.species->spe / 2) * 2, 1, 255) / 2;
        bool crit = crit_chance > 0 && rng.chance(crit_chance, 256);
        bool physical = gen1_is_physical(mv->type);
        int atk = physical ? a.atk : a.spc;
        int def = physical ? d.def : d.spc;
        // Gen 1: a crit ignores the burn Attack-drop and screens (uses the unmodified stats).
        if (!crit) {
            if (physical && a.status == Status::Burn) atk = std::max(1, atk / 2);  // burn halves Atk
            if (physical && d.reflect) def *= 2;                                   // Reflect doubles Def
        }
        // Gen 1 stat rollover: if attack or defense >= 256, divide BOTH by 4 (mod 256).
        if (atk >= 256 || def >= 256) {
            atk = std::max(1, (atk / 4) % 256);
            def = (def / 4) % 256;
            if (def == 0) def = 1;
        }
        if (mv->effect == Effect::SelfDestruct) def = std::max(1, def / 2);  // Explosion halves Def
        d.hp -= gen1_damage(a.level, mv->power, atk, def, stab, mult, crit, rng.range(217, 255));
    }

    if (mv->effect == Effect::SelfDestruct) a.hp = 0;
    apply_effect(mv, a, d, rng);
}

void try_move(Side& as, Side& ds, int moveidx, RNG& rng) {
    if (as.mon().fainted()) return;
    if (!can_act(as.mon(), rng)) return;
    use_move(as, ds, moveidx, rng);
}

void residual(Pokemon& p) {
    if (p.fainted()) return;
    if (p.status == Status::Burn || p.status == Status::Poison)
        p.hp -= std::max(1, p.max_hp / 16);
}

void do_switch(Side& s, int idx) {
    s.mon().reflect = false;  // Reflect ends when its user leaves the field
    s.active = idx;
}
}  // namespace

Result Battle::result() const {
    bool d1 = p1.all_fainted(), d2 = p2.all_fainted();
    if (d1 && d2) return Result::Tie;
    if (d2) return Result::P1Win;
    if (d1) return Result::P2Win;
    return Result::Ongoing;
}

std::vector<Choice> Battle::choices(int player) const {
    const Side& s = (player == 0) ? p1 : p2;
    std::vector<Choice> out;
    if (s.all_fainted()) return out;
    auto bench = [&]() {
        for (int i = 0; i < static_cast<int>(s.team.size()); ++i)
            if (i != s.active && !s.team[i].fainted()) out.push_back({ChoiceKind::Switch, i});
    };
    if (s.must_switch) {
        bench();
        return out;
    }
    for (int i = 0; i < static_cast<int>(s.mon().moves.size()); ++i)
        out.push_back({ChoiceKind::Move, i});
    bench();
    return out;
}

Result Battle::step(const Choice& c1, const Choice& c2) {
    if (p1.must_switch || p2.must_switch) {
        if (p1.must_switch && c1.kind == ChoiceKind::Switch) do_switch(p1, c1.index);
        if (p2.must_switch && c2.kind == ChoiceKind::Switch) do_switch(p2, c2.index);
        p1.must_switch = p1.mon().fainted() && p1.has_alive_bench();
        p2.must_switch = p2.mon().fainted() && p2.has_alive_bench();
        return result();
    }

    ++turn;

    if (c1.kind == ChoiceKind::Switch) do_switch(p1, c1.index);
    if (c2.kind == ChoiceKind::Switch) do_switch(p2, c2.index);

    bool m1 = c1.kind == ChoiceKind::Move;
    bool m2 = c2.kind == ChoiceKind::Move;
    int s1 = effective_speed(p1.mon()), s2 = effective_speed(p2.mon());
    bool p1_first = (s1 != s2) ? (s1 > s2) : rng.chance(1, 2);

    auto act1 = [&]() { if (m1) try_move(p1, p2, c1.index, rng); };
    auto act2 = [&]() { if (m2) try_move(p2, p1, c2.index, rng); };
    if (p1_first) { act1(); act2(); } else { act2(); act1(); }

    residual(p1.mon());
    residual(p2.mon());

    p1.must_switch = p1.mon().fainted() && p1.has_alive_bench();
    p2.must_switch = p2.mon().fainted() && p2.has_alive_bench();

    return result();
}

Battle make_battle(const TeamSpec& team1, const TeamSpec& team2, uint64_t seed) {
    auto build = [](const TeamSpec& spec) {
        Side s;
        for (const auto& entry : spec) {
            std::vector<const MoveData*> moves;
            for (const auto& mn : entry.second) moves.push_back(&move(mn));
            s.team.push_back(make_pokemon(&species(entry.first), std::move(moves)));
        }
        return s;
    };
    return Battle(build(team1), build(team2), seed);
}

}  // namespace cinnabar
