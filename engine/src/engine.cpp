#include "cinnabar/engine.hpp"

#include "cinnabar/gen1_data.hpp"  // generated from Showdown: GEN1_TYPE_CHART, GEN1_SPECIES

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#ifdef CINNABAR_DEBUG
#include <cstdio>
#endif

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
        // Psychic: 33% chance to lower the target's Special by 1 stage (Gen 1 secondary).
        add({"Psychic", Type::Psychic, Category::Special, 90, 100, 0, Effect::None, 0, 2, -1, true, 33});
        add({"Amnesia", Type::Psychic, Category::Status, 0, 100, 0, Effect::None, 0, 2, 2, false, 0});
        add({"Swords Dance", Type::Normal, Category::Status, 0, 100, 0, Effect::None, 0, 0, 2, false, 0});
        add({"Agility", Type::Psychic, Category::Status, 0, 100, 0, Effect::None, 0, 3, 2, false, 0});
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

int gen1_damage(int level, int power, int attack, int defense, bool stab,
                Type move_type, Type def_t1, Type def_t2, bool crit, int random) {
    if (power <= 0) return 0;
    int L = crit ? 2 * level : level;
    long dmg = (2L * L) / 5 + 2;
    dmg = dmg * power * attack / std::max(1, defense);
    dmg = dmg / 50;
    if (dmg > 997) dmg = 997;  // Showdown clampIntRange(floor(dmg/50), 0, 997)
    dmg += 2;
    if (stab) dmg += dmg / 2;  // STAB: damage += floor(damage / 2)
    // Type effectiveness per defending type, in order, flooring after each (Showdown's
    // ×20/10 / ×5/10 steps) — the order matters for dual types (e.g. Ice vs Water/Flying).
    auto apply_type = [&](Type dt) {
        if (dt == Type::None) return;
        double e = type_effectiveness(move_type, dt);
        if (e == 0.0) dmg = 0;            // immune (normally filtered earlier in use_move)
        else if (e > 1.0) dmg = dmg * 20 / 10;
        else if (e < 1.0) dmg = dmg * 5 / 10;
    };
    apply_type(def_t1);
    apply_type(def_t2);
    if (dmg == 0) return 0;
    if (dmg > 1) dmg = dmg * random / 255;
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
    p.m_atk = p.atk;  // modifiedStats start equal to stored stats
    p.m_def = p.def;
    p.m_spc = p.spc;
    p.m_spe = p.spe;
    p.moves = std::move(moves);
    return p;
}

uint32_t RNG::next() {
    // Showdown Gen5RNG: x = x * a + c (mod 2^64); output is the upper 32 bits.
    state = state * 0x5D588B656C078965ULL + 0x00269EC3ULL;
    return static_cast<uint32_t>(state >> 32);
}
int RNG::random(int n) {
    int r = static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(n)) >> 32);
#ifdef CINNABAR_DEBUG
    std::fprintf(stderr, "  ourrng random(%d) = %d\n", n, r);
#endif
    return r;
}
int RNG::random(int from, int to) {
    int r = static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(to - from)) >> 32) + from;
#ifdef CINNABAR_DEBUG
    std::fprintf(stderr, "  ourrng random(%d,%d) = %d\n", from, to, r);
#endif
    return r;
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
    return p.m_spe;  // paralysis (÷4) and Agility are baked into the modified Speed stat
}

// Gen 1 stat-stage machinery. Stored stats are immutable; modifiedStats carry boosts plus
// the burn/paralysis drops. boostBy resets modifiedStats from stored then applies the stage
// multiplier — which is why it *discards* an existing burn/paralysis drop (a real Gen 1 bug).
const double BOOST_POS[7] = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
const int BOOST_NEG[7] = {100, 66, 50, 40, 33, 28, 25};

int& stored_ref(Pokemon& p, int s) {
    switch (s) { case 0: return p.atk; case 1: return p.def; case 2: return p.spc; default: return p.spe; }
}
int& mod_ref(Pokemon& p, int s) {
    switch (s) { case 0: return p.m_atk; case 1: return p.m_def; case 2: return p.m_spc; default: return p.m_spe; }
}
int& stage_ref(Pokemon& p, int s) {
    switch (s) { case 0: return p.boost_atk; case 1: return p.boost_def; case 2: return p.boost_spc; default: return p.boost_spe; }
}

// Recompute modifiedStats[s] from stored stat + current stage (Showdown's boostBy path).
void recompute_stat(Pokemon& p, int s) {
    int stored = stored_ref(p, s), stage = stage_ref(p, s);
    double v = stage >= 0 ? stored * BOOST_POS[stage] : stored * (BOOST_NEG[-stage] / 100.0);
    long iv = static_cast<long>(std::floor(v));
    iv = std::clamp<long>(iv, 1, 999);
    mod_ref(p, s) = static_cast<int>(iv);
}

// modifyStat: multiply the *current* modifiedStat (burn ×0.5 Atk, paralysis ×0.25 Spe), min 1.
void modify_stat(Pokemon& p, int s, double mult) {
    long v = static_cast<long>(std::floor(mod_ref(p, s) * mult));
    mod_ref(p, s) = static_cast<int>(std::max<long>(1, v));
}

void do_boost(Pokemon& p, int s, int delta) {
    int& st = stage_ref(p, s);
    st = std::clamp(st + delta, -6, 6);
    recompute_stat(p, s);
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
            return !rng.chance(63, 256);  // Showdown gen1: 63/256 chance of full paralysis
        default:
            return true;
    }
}

void apply_effect(const MoveData* mv, Pokemon& user, Pokemon& tgt, RNG& rng) {
    if (mv->effect == Effect::None) return;
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

    // Status-inflicting effects on the target.
    if (tgt.fainted()) return;  // KO'd by the damage -> no status roll (Showdown guards target.hp>0)

    const bool secondary = mv->effect_chance < 100;  // <100 == a damaging move's secondary
    const bool par_brn_frz = mv->effect == Effect::Paralyze || mv->effect == Effect::Burn ||
                             mv->effect == Effect::Freeze;

    if (secondary) {
        // Gen 1: a secondary par/brn/frz never triggers on a target sharing the move's type,
        // and the RNG is NOT rolled in that case (so draw counts stay aligned with Showdown).
        if (par_brn_frz && (mv->type == tgt.species->t1 || mv->type == tgt.species->t2)) return;
        // Showdown rolls randomChance(ceil(chance*256/100), 256). The roll is spent even if the
        // target is already statused (the set-status just fails afterwards).
        const int num = (mv->effect_chance * 256 + 99) / 100;  // ceil(chance * 256 / 100)
        if (!rng.chance(num, 256)) return;
    } else {
        // Primary status move (Thunder Wave, Sleep Powder, ...): guaranteed on hit, no roll,
        // but blocked by type immunity (e.g. Thunder Wave vs a Ground-type).
        if (mv->power == 0 && mv->fixed == 0 && combined_effectiveness(mv->type, tgt.species) == 0.0)
            return;
    }

    if (tgt.status != Status::None) return;  // already statused -> set-status fails (roll spent)

    switch (mv->effect) {
        case Effect::Paralyze: tgt.status = Status::Paralysis; modify_stat(tgt, 3, 0.25); break;
        case Effect::Sleep:    tgt.status = Status::Sleep; tgt.sleep_turns = rng.range(1, 7); break;
        case Effect::Freeze:   tgt.status = Status::Freeze; break;
        case Effect::Burn:     tgt.status = Status::Burn; modify_stat(tgt, 0, 0.5); break;
        case Effect::Poison:   tgt.status = Status::Poison; break;
        default: break;
    }
}

// Stat-stage changes: self boosts (Amnesia/Swords Dance/Agility) or a damaging move's foe
// secondary (Psychic's −Special). Mirrors Showdown's moveData.boosts block, including the
// quirk that applying boosts re-applies the user's foe's paralysis/burn stat drop.
void apply_boosts(const MoveData* mv, Pokemon& user, Pokemon& foe, RNG& rng) {
    if (mv->boost_stat < 0 || mv->boost_stat > 3) return;  // only atk/def/spc/spe modelled
    Pokemon& tgt = mv->boost_target_foe ? foe : user;
    if (mv->boost_target_foe) {
        if (tgt.fainted()) return;  // secondary needs a surviving target
        if (mv->boost_chance < 100) {
            int num = (mv->boost_chance * 256 + 99) / 100;  // ceil(chance * 256 / 100)
            if (!rng.chance(num, 256)) return;
        }
    }
    do_boost(tgt, mv->boost_stat, mv->boost_stages);
    if (foe.status == Status::Paralysis) modify_stat(foe, 3, 0.25);  // Gen 1 re-application quirk
    if (foe.status == Status::Burn) modify_stat(foe, 0, 0.5);
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
                          mv->effect == Effect::Reflect ||
                          (mv->boost_stat >= 0 && !mv->boost_target_foe);  // Amnesia/SD/Agility
    if (!self_targeting && mv->accuracy > 0) {
        int acc = std::clamp(mv->accuracy * 255 / 100, 1, 255);
        if (!rng.chance(acc, 256)) return;  // miss (includes the gen1 1/256)
    }

    if (mv->fixed > 0) {
        if (combined_effectiveness(mv->type, d.species) == 0.0) return;
        d.hp -= mv->fixed;
        if (d.hp < 0) d.hp = 0;
    } else if (mv->power > 0 && !d.fainted()) {
        double mult = combined_effectiveness(mv->type, d.species);
        if (mult == 0.0) return;  // immune: no damage, no secondary
        bool stab = (mv->type == a.species->t1 || mv->type == a.species->t2);
        // Crit — Showdown gen1 derives the chance from BASE species Speed, not the live stat:
        // critChance = clamp(floor(baseSpe/2)*2, 1, 255), then /2 for a normal crit ratio.
        int crit_chance = std::clamp((a.species->spe / 2) * 2, 1, 255) / 2;
        bool crit = crit_chance > 0 && rng.chance(crit_chance, 256);
        bool physical = gen1_is_physical(mv->type);
        int atk, def;
        if (crit) {                              // a crit ignores boosts, burn, and screens
            atk = physical ? a.atk : a.spc;      // stored (unmodified) stats
            def = physical ? d.def : d.spc;
        } else {
            atk = physical ? a.m_atk : a.m_spc;  // modifiedStats (boosts + burn drop)
            def = physical ? d.m_def : d.m_spc;
            if (physical && d.reflect) def *= 2;  // Reflect doubles Def (a screen, not on crit)
        }
        // Gen 1 stat rollover: if attack or defense >= 256, divide BOTH by 4 (mod 256).
        if (atk >= 256 || def >= 256) {
            atk = std::max(1, (atk / 4) % 256);
            def = (def / 4) % 256;
            if (def == 0) def = 1;
        }
        if (mv->effect == Effect::SelfDestruct) def = std::max(1, def / 2);  // Explosion halves Def
        int roll = rng.range(217, 255);
        int dmg = gen1_damage(a.level, mv->power, atk, def, stab, mv->type,
                              d.species->t1, d.species->t2, crit, roll);
#ifdef CINNABAR_DEBUG
        std::fprintf(stderr, "[hit] %s->%s atk=%d def=%d stab=%d mult=%.2f crit=%d roll=%d dmg=%d\n",
                     a.species->name.c_str(), d.species->name.c_str(), atk, def,
                     static_cast<int>(stab), mult, static_cast<int>(crit), roll, dmg);
#endif
        d.hp -= dmg;
        if (d.hp < 0) d.hp = 0;  // Showdown caps damage at remaining HP (fainted sits at 0)
    }

    if (mv->effect == Effect::SelfDestruct) a.hp = 0;
    apply_effect(mv, a, d, rng);
    apply_boosts(mv, a, d, rng);
}

void try_move(Side& as, Side& ds, int moveidx, RNG& rng) {
    if (as.mon().fainted()) return;
    if (!can_act(as.mon(), rng)) return;
    use_move(as, ds, moveidx, rng);
}

void residual(Pokemon& p) {
    if (p.fainted()) return;
    if (p.status == Status::Burn || p.status == Status::Poison) {
        p.hp -= std::max(1, p.max_hp / 16);
        if (p.hp < 0) p.hp = 0;
    }
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

    // Speed ties: Showdown burns extra random(0,2) "speed-tie shuffles" across its scheduler
    // (it re-shuffles the equal-speed actives in every event pass). To stay bit-aligned we
    // replicate the steady-state frame — both actives present and acting: 3 shuffles before
    // the moves (the first decides order), 1 between the moves, 2 after — only the first
    // changes observable state. (Distinct speeds consume nothing, matching Showdown.)
    bool tie = (m1 && m2 && s1 == s2);
    bool p1_first;
    if (s1 != s2) {
        p1_first = s1 > s2;
    } else if (tie) {
        p1_first = (rng.random(0, 2) == 0);  // move-order shuffle: 0 = no swap = p1 first
        rng.random(0, 2);
        rng.random(0, 2);
    } else {
        p1_first = (rng.random(0, 2) == 0);
    }
#ifdef CINNABAR_DEBUG
    std::fprintf(stderr, "[turn %d] s1=%d s2=%d p1_first=%d tie=%d\n", turn, s1, s2,
                 static_cast<int>(p1_first), static_cast<int>(tie));
#endif

    auto act1 = [&]() { if (m1) try_move(p1, p2, c1.index, rng); };
    auto act2 = [&]() { if (m2) try_move(p2, p1, c2.index, rng); };
    if (p1_first) { act1(); if (tie) rng.random(0, 2); act2(); }
    else          { act2(); if (tie) rng.random(0, 2); act1(); }
    if (tie) { rng.random(0, 2); rng.random(0, 2); }

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
    Battle b(build(team1), build(team2), seed);
    // Battle-start speed-tie shuffles: Showdown shuffles the equal-speed actives four times
    // during its start sequence before turn 1. Replicate so the RNG stream stays aligned.
    if (b.p1.mon().m_spe == b.p2.mon().m_spe) {
        for (int i = 0; i < 4; ++i) b.rng.random(0, 2);
    }
    return b;
}

}  // namespace cinnabar
