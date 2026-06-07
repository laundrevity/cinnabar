#include "cinnabar/engine.hpp"

#include <algorithm>

namespace cinnabar {

namespace {
constexpr int N = 15;  // real types (excludes None)
int CHART[N][N];
bool chart_ready = false;

void se(int a, int d) { CHART[a][d] = 200; }
void nve(int a, int d) { CHART[a][d] = 50; }
void imm(int a, int d) { CHART[a][d] = 0; }

void build_chart() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) CHART[i][j] = 100;
    using T = Type;
    auto A = [](T t) { return static_cast<int>(t); };

    nve(A(T::Normal), A(T::Rock));      imm(A(T::Normal), A(T::Ghost));
    se(A(T::Fighting), A(T::Normal));   se(A(T::Fighting), A(T::Rock));   se(A(T::Fighting), A(T::Ice));
    nve(A(T::Fighting), A(T::Flying));  nve(A(T::Fighting), A(T::Poison)); nve(A(T::Fighting), A(T::Bug));
    nve(A(T::Fighting), A(T::Psychic)); imm(A(T::Fighting), A(T::Ghost));
    se(A(T::Flying), A(T::Fighting));   se(A(T::Flying), A(T::Bug));      se(A(T::Flying), A(T::Grass));
    nve(A(T::Flying), A(T::Rock));      nve(A(T::Flying), A(T::Electric));
    se(A(T::Poison), A(T::Bug));        se(A(T::Poison), A(T::Grass));    // Gen 1: Poison SE vs Bug
    nve(A(T::Poison), A(T::Poison));    nve(A(T::Poison), A(T::Ground));  nve(A(T::Poison), A(T::Rock));
    nve(A(T::Poison), A(T::Ghost));
    se(A(T::Ground), A(T::Poison));     se(A(T::Ground), A(T::Rock));     se(A(T::Ground), A(T::Fire));
    se(A(T::Ground), A(T::Electric));   nve(A(T::Ground), A(T::Grass));   nve(A(T::Ground), A(T::Bug));
    imm(A(T::Ground), A(T::Flying));
    se(A(T::Rock), A(T::Flying));       se(A(T::Rock), A(T::Bug));        se(A(T::Rock), A(T::Fire));
    se(A(T::Rock), A(T::Ice));          nve(A(T::Rock), A(T::Fighting));  nve(A(T::Rock), A(T::Ground));
    se(A(T::Bug), A(T::Poison));        se(A(T::Bug), A(T::Grass));       se(A(T::Bug), A(T::Psychic)); // Gen 1: Bug SE vs Poison
    nve(A(T::Bug), A(T::Fighting));     nve(A(T::Bug), A(T::Flying));     nve(A(T::Bug), A(T::Ghost));
    nve(A(T::Bug), A(T::Fire));
    se(A(T::Ghost), A(T::Ghost));       imm(A(T::Ghost), A(T::Normal));   imm(A(T::Ghost), A(T::Psychic)); // Gen 1 bug
    se(A(T::Fire), A(T::Bug));          se(A(T::Fire), A(T::Grass));      se(A(T::Fire), A(T::Ice));
    nve(A(T::Fire), A(T::Rock));        nve(A(T::Fire), A(T::Fire));      nve(A(T::Fire), A(T::Water));
    nve(A(T::Fire), A(T::Dragon));
    se(A(T::Water), A(T::Ground));      se(A(T::Water), A(T::Rock));      se(A(T::Water), A(T::Fire));
    nve(A(T::Water), A(T::Water));      nve(A(T::Water), A(T::Grass));    nve(A(T::Water), A(T::Dragon));
    se(A(T::Grass), A(T::Ground));      se(A(T::Grass), A(T::Rock));      se(A(T::Grass), A(T::Water));
    nve(A(T::Grass), A(T::Flying));     nve(A(T::Grass), A(T::Poison));   nve(A(T::Grass), A(T::Bug));
    nve(A(T::Grass), A(T::Fire));       nve(A(T::Grass), A(T::Grass));    nve(A(T::Grass), A(T::Dragon));
    se(A(T::Electric), A(T::Flying));   se(A(T::Electric), A(T::Water));  nve(A(T::Electric), A(T::Grass));
    nve(A(T::Electric), A(T::Electric)); nve(A(T::Electric), A(T::Dragon)); imm(A(T::Electric), A(T::Ground));
    se(A(T::Psychic), A(T::Fighting));  se(A(T::Psychic), A(T::Poison));  nve(A(T::Psychic), A(T::Psychic));
    se(A(T::Ice), A(T::Flying));        se(A(T::Ice), A(T::Ground));      se(A(T::Ice), A(T::Grass));
    se(A(T::Ice), A(T::Dragon));        nve(A(T::Ice), A(T::Water));      nve(A(T::Ice), A(T::Ice));
    se(A(T::Dragon), A(T::Dragon));
    chart_ready = true;
}
}  // namespace

double type_effectiveness(Type attacking, Type defending) {
    if (attacking == Type::None || defending == Type::None) return 1.0;
    if (!chart_ready) build_chart();
    return CHART[static_cast<int>(attacking)][static_cast<int>(defending)] / 100.0;
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
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<uint32_t>(state >> 32);
}
int RNG::range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
bool RNG::chance(int num, int den) { return static_cast<int>(next() % static_cast<uint32_t>(den)) < num; }

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

// Can this Pokémon act this turn? Mutates sleep counter / thaws as a side effect.
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
            return;  // damage + user faint handled by use_move
        default:
            break;  // status-inflicting effects below
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

    if (mv->accuracy < 100 && rng.range(1, 100) > mv->accuracy) return;  // miss

    if (mv->fixed > 0) {
        if (combined_effectiveness(mv->type, d.species) == 0.0) return;
        d.hp -= mv->fixed;
    } else if (mv->power > 0 && !d.fainted()) {
        double mult = combined_effectiveness(mv->type, d.species);
        if (mult == 0.0) return;  // immune: no damage, no secondary
        bool stab = (mv->type == a.species->t1 || mv->type == a.species->t2);
        bool crit = rng.chance(a.species->spe, 512);
        bool physical = gen1_is_physical(mv->type);
        int atk = physical ? a.atk : a.spc;
        int def = physical ? d.def : d.spc;
        if (physical && d.reflect) def *= 2;                                  // Reflect
        if (mv->effect == Effect::SelfDestruct) def = std::max(1, def / 2);   // Explosion halves Def
        if (physical && a.status == Status::Burn) atk = std::max(1, atk / 2);  // burn halves Attack
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
    // Forced-switch (replacement) step: only switches resolve, no turn advances.
    if (p1.must_switch || p2.must_switch) {
        if (p1.must_switch && c1.kind == ChoiceKind::Switch) do_switch(p1, c1.index);
        if (p2.must_switch && c2.kind == ChoiceKind::Switch) do_switch(p2, c2.index);
        p1.must_switch = p1.mon().fainted() && p1.has_alive_bench();
        p2.must_switch = p2.mon().fainted() && p2.has_alive_bench();
        return result();
    }

    ++turn;

    // 1) Switches resolve before any move.
    if (c1.kind == ChoiceKind::Switch) do_switch(p1, c1.index);
    if (c2.kind == ChoiceKind::Switch) do_switch(p2, c2.index);

    // 2) Moves, in Speed order (ties broken randomly).
    bool m1 = c1.kind == ChoiceKind::Move;
    bool m2 = c2.kind == ChoiceKind::Move;
    int s1 = effective_speed(p1.mon()), s2 = effective_speed(p2.mon());
    bool p1_first = (s1 != s2) ? (s1 > s2) : rng.chance(1, 2);

    auto act1 = [&]() { if (m1) try_move(p1, p2, c1.index, rng); };
    auto act2 = [&]() { if (m2) try_move(p2, p1, c2.index, rng); };
    if (p1_first) { act1(); act2(); } else { act2(); act1(); }

    // 3) End-of-turn residual damage.
    residual(p1.mon());
    residual(p2.mon());

    // 4) Flag forced switches for fainted actives that still have a bench.
    p1.must_switch = p1.mon().fainted() && p1.has_alive_bench();
    p2.must_switch = p2.mon().fainted() && p2.has_alive_bench();

    return result();
}

}  // namespace cinnabar
