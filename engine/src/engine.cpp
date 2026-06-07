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
    // Built from the generated GEN1_MOVES table (engine/tools/gen_data.py from Showdown).
    static const std::unordered_map<std::string, MoveData> table = [] {
        std::unordered_map<std::string, MoveData> m;
        for (const auto& e : GEN1_MOVES) {
            m.emplace(e.name, MoveData{e.name, e.type, e.category, e.power, e.accuracy, e.fixed,
                                       e.effect, e.effect_chance, e.boost_stat, e.boost_stages,
                                       e.boost_target_foe, e.boost_chance, e.high_crit, e.pp,
                                       e.recharge, e.recoil_num, e.recoil_den, e.ignore_immunity,
                                       e.priority, e.skip_lastdamage});
        }
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
    // PP per slot: Showdown's max = base * 8/5 (3 PP ups). Untracked moves (pp 0) -> unlimited.
    for (const MoveData* mv : p.moves)
        p.pp.push_back(mv && mv->pp > 0 ? mv->pp * 8 / 5 : -1);
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
    // Showdown gen1 onBeforeMove priority order: frz(12) > slp(10) > mustrecharge(7) > par(2).
    // The order matters for RNG alignment: recharge cancels the move *before* paralysis rolls,
    // so a recharge turn must NOT consume the 63/256 full-paralysis draw.
    switch (p.status) {
        case Status::Freeze:
            return false;  // Gen 1: frozen solid (thaw not modelled yet)
        case Status::Sleep:
            if (p.sleep_turns > 0) --p.sleep_turns;
            if (p.sleep_turns <= 0) p.status = Status::None;
            return false;  // Gen 1: a turn is always lost to sleep, including the wake turn
        default:
            break;
    }
    if (p.must_recharge) {        // Hyper Beam recharge: spend this turn doing nothing, clear the
        p.must_recharge = false;  // flag, and skip the paralysis roll (priority 7 > par's 2).
        return false;
    }
    // Confusion (onBeforeMove priority 3, after recharge, before paralysis). Decrement; if it just
    // wore off the mon acts (and paralysis still rolls); else 50% it hits itself and loses the turn.
    if (p.confuse_turns > 0) {
        --p.confuse_turns;
        if (p.confuse_turns > 0 && !rng.chance(128, 256)) {  // 50%: hurt itself in confusion
            // Typeless 40-BP physical using the user's own (modified) Atk/Def — no type/STAB/crit.
            long a = (2L * p.level) / 5 + 2;
            long dmg = a * p.m_atk * 40 / std::max(1, p.m_def);
            p.hp -= static_cast<int>(dmg / 50 + 2);
            if (p.hp < 0) p.hp = 0;
            return false;  // move cancelled by the self-hit
        }
    }
    if (p.status == Status::Paralysis)
        return !rng.chance(63, 256);  // Showdown gen1: 63/256 chance of full paralysis
    return true;
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
        case Effect::Substitute:
            // Gen 1: fails if a sub is already up or HP < maxhp/4. Otherwise pay floor(maxhp/4)
            // and create a sub with floor(maxhp/4)+1 HP. (maxhp<=3 makes no sub — Showdown quirk.)
            if (user.has_substitute) return;
            if (user.hp * 4 < user.max_hp) return;
            if (user.max_hp > 3) {
                int cost = user.max_hp / 4;
                user.hp -= cost;
                if (user.hp > 0) { user.has_substitute = true; user.sub_hp = cost + 1; }
                else user.hp = 0;
            }
            return;
        default:
            break;
    }

    // Status-inflicting effects on the target.
    if (tgt.fainted()) return;  // KO'd by the damage -> no status roll (Showdown guards target.hp>0)

    if (mv->effect == Effect::Confuse) {  // Confuse Ray: a volatile, independent of major status
        if (tgt.has_substitute) return;  // a sub blocks confusion
        // Gen 1: confusion ignores type immunity — Confuse Ray confuses Normal types despite being Ghost.
        if (tgt.confuse_turns == 0) tgt.confuse_turns = rng.range(2, 5);  // 2-5 turns (else already confused)
        return;
    }

    const bool secondary = mv->effect_chance < 100;  // <100 == a damaging move's secondary
    // Substitute (Gen 1): while the target's sub is up, a damaging move's secondary status is
    // skipped entirely (no RNG roll); a primary status move is blocked only for poison —
    // paralysis / sleep / freeze / burn pass through the substitute.
    if (tgt.has_substitute) {
        if (secondary) return;
        if (mv->effect == Effect::Poison) return;
    }
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
    }
    // A primary status move applies guaranteed on hit. Type immunity was already handled before
    // the accuracy roll (in use_move), so anything reaching here is either not immune or a move
    // that ignores immunity (Confuse Ray, Glare).

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
        if (tgt.has_substitute) return;  // Gen 1: a sub blocks foe stat-drops (no roll spent)
        if (mv->boost_chance > 0 && mv->boost_chance < 100) {  // a damaging move's % secondary
            int num = (mv->boost_chance * 256 + 99) / 100;     // ceil(chance * 256 / 100)
            if (!rng.chance(num, 256)) return;
        }
        // boost_chance 0 (a foe-targeting status move like Growl/Screech) or >=100: always apply.
    }
    do_boost(tgt, mv->boost_stat, mv->boost_stages);
    if (foe.status == Status::Paralysis) modify_stat(foe, 3, 0.25);  // Gen 1 re-application quirk
    if (foe.status == Status::Burn) modify_stat(foe, 0, 0.5);
}

// Struggle: used when every move is out of PP. Gen 1 — Normal, 50 BP, accuracy 100 (rolls the
// 1/256 miss like any move), 1/2-damage recoil.
static const MoveData STRUGGLE{.name = "Struggle", .type = Type::Normal, .category = Category::Physical,
                               .power = 50, .accuracy = 100, .recoil_num = 1, .recoil_den = 2};

void use_move(Side& as, Side& ds, int moveidx, RNG& rng, int& last_damage) {
    Pokemon& a = as.mon();
    Pokemon& d = ds.mon();
    const bool struggle = (moveidx == -1);
    const MoveData* mv;
    if (struggle) {
        mv = &STRUGGLE;
    } else {
        if (moveidx < 0 || moveidx >= static_cast<int>(a.moves.size())) return;
        mv = a.moves[moveidx];
        if (!mv) return;
        if (a.pp[moveidx] > 0) --a.pp[moveidx];  // deduct PP on use (gen1: even if it misses/fails)
    }
    as.last_move = mv;  // record the move used (Counter reads the opponent's last used move)
    if (!mv->skip_lastdamage) last_damage = 0;  // damaging non-Counter moves clear it; Counter/status don't
    int recoil_base = 0;     // damage the recoil is computed from (capped; uncapped vs a sub)
    bool recoil_ok = false;  // recoil applies (false vs a sub that broke on this hit)

    // Moves that target the user (Recover/Rest/Reflect/Substitute, self-boosts) skip the
    // accuracy roll and the immunity check below.
    bool self_targeting = mv->effect == Effect::Heal || mv->effect == Effect::Rest ||
                          mv->effect == Effect::Reflect || mv->effect == Effect::Substitute ||
                          (mv->boost_stat >= 0 && !mv->boost_target_foe);  // Amnesia/SD/Agility

    // Gen 1 checks type immunity BEFORE accuracy (scripts.ts: runImmunity precedes the accuracy
    // roll), so an immune move never rolls the 1/256. Only moves that do NOT ignore immunity are
    // blocked: damaging moves and Thunder Wave respect it; most status moves (Confuse Ray, Glare)
    // set ignoreImmunity and pass through.
    if (!self_targeting && !mv->ignore_immunity &&
        combined_effectiveness(mv->type, d.species) == 0.0) {
        if (mv->effect == Effect::SelfDestruct) a.hp = 0;  // Explosion still faints the user
        return;
    }

    // Accuracy — Showdown gen1 rolls randomChance(clamp(floor(acc*255/100),1,255), 256) for
    // every non-self-targeting move, *including* 100%-accuracy ones (the 1/256 miss).
    if (!self_targeting && mv->accuracy > 0) {
        int acc = std::clamp(mv->accuracy * 255 / 100, 1, 255);
        if (!rng.chance(acc, 256)) {  // miss (includes the gen1 1/256)
            if (mv->effect == Effect::SelfDestruct) a.hp = 0;  // gen1: Explosion faints user on a miss
            last_damage = 0;  // a miss clears last_damage (Counter can't reflect a missed hit)
            return;
        }
    }

    if (mv->effect == Effect::Counter) {
        // Gen 1 Counter: reflect 2x the battle's last damage, but only if BOTH the opponent's last
        // *used* and last *selected* moves are "counterable" (Normal/Fighting, BP>0, not Counter).
        // Exactly one counterable -> Showdown's Desync Clause Mod fails it; neither -> fail.
        auto counterable = [](const MoveData* m) {
            return m && m->power > 0 && m->effect != Effect::Counter &&
                   (m->type == Type::Normal || m->type == Type::Fighting);
        };
        bool lu = counterable(ds.last_move), ls = counterable(ds.last_selected);
        if ((lu || ls) && last_damage > 0 && lu == ls) {
            int dmg = 2 * last_damage;  // typeless, ignores crit; hits the sub like other damage
            if (d.has_substitute) {
                d.sub_hp -= dmg > d.sub_hp ? d.sub_hp : dmg;
                if (d.sub_hp <= 0) { d.has_substitute = false; d.sub_hp = 0; }
            } else {
                dmg = dmg < d.hp ? dmg : d.hp;
                d.hp -= dmg;
            }
            last_damage = dmg;
        }
        return;
    }

    if (mv->fixed > 0) {
        if (combined_effectiveness(mv->type, d.species) == 0.0) return;
        if (d.has_substitute) {  // fixed damage hits the sub (no overflow to real HP)
            d.sub_hp -= mv->fixed > d.sub_hp ? d.sub_hp : mv->fixed;
            if (d.sub_hp <= 0) { d.has_substitute = false; d.sub_hp = 0; }
            last_damage = mv->fixed;  // uncapped vs a sub
        } else {
            int hit = mv->fixed < d.hp ? mv->fixed : d.hp;
            d.hp -= hit;
            last_damage = hit;
        }
    } else if (mv->power > 0 && !d.fainted()) {
        double mult = combined_effectiveness(mv->type, d.species);
        if (mult == 0.0) {  // immune: no damage, no secondary
            if (mv->effect == Effect::SelfDestruct) a.hp = 0;  // ...but Explosion still faints the user
            return;
        }
        bool stab = (mv->type == a.species->t1 || mv->type == a.species->t2);
        // Crit — Showdown gen1 derives the chance from BASE species Speed, not the live stat:
        // base = clamp(floor(baseSpe/2)*2, 1, 255); then /2 (normal) or *4 clamp (high crit ratio).
        int crit_chance = std::clamp((a.species->spe / 2) * 2, 1, 255);
        crit_chance = mv->high_crit ? std::clamp(crit_chance * 4, 1, 255) : crit_chance / 2;
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
        if (d.has_substitute) {  // Gen 1: damage hits the sub; excess is NOT dealt to real HP
            d.sub_hp -= dmg > d.sub_hp ? d.sub_hp : dmg;
            recoil_ok = d.sub_hp > 0;  // recoil happens only if the sub survived (Gen 1)
            if (d.sub_hp <= 0) { d.has_substitute = false; d.sub_hp = 0; }
            recoil_base = dmg;  // uncapped (the Gen 1 sub quirk)
        } else {
            int actual = dmg < d.hp ? dmg : d.hp;  // Showdown caps damage at the target's HP
            d.hp -= actual;
            recoil_base = actual;
            recoil_ok = true;
        }
        last_damage = recoil_base;  // what a subsequent Counter would double
    }

    if (mv->effect == Effect::SelfDestruct) a.hp = 0;
    // Recoil to the user (Double-Edge/Take Down/Submission/Struggle): floor(dmg * num/den), min 1.
    // Vs a Substitute, Gen 1 deals recoil only if the sub survived, off the uncapped damage.
    if (mv->recoil_den > 0 && recoil_ok && recoil_base > 0) {
        a.hp -= std::max(1, recoil_base * mv->recoil_num / mv->recoil_den);
        if (a.hp < 0) a.hp = 0;
    }
    apply_effect(mv, a, d, rng);
    apply_boosts(mv, a, d, rng);
    // Hyper Beam: the user owes a recharge turn — UNLESS the move KO'd the target (the famous
    // Gen 1 "no recharge on KO"). Reaching here means the move hit (a miss/immunity returned early).
    if (mv->recharge && !d.fainted()) a.must_recharge = true;
}

void try_move(Side& as, Side& ds, int moveidx, RNG& rng, int& last_damage) {
    if (as.mon().fainted()) return;
    if (!can_act(as.mon(), rng)) return;  // sleep/freeze/recharge/full-para/confusion self-hit
    use_move(as, ds, moveidx, rng, last_damage);
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
    Pokemon& in = s.mon();
    in.must_recharge = false;  // volatiles clear on switch (a recharge can't carry to a new mon)
    in.has_substitute = false; in.sub_hp = 0;  // a Substitute does not persist across a switch
    in.confuse_turns = 0;      // confusion clears on switch out
    // Gen 1: stat stages reset on switch — recompute modified stats from the stored stats...
    in.boost_atk = in.boost_def = in.boost_spc = in.boost_spe = 0;
    in.m_atk = in.atk; in.m_def = in.def; in.m_spc = in.spc; in.m_spe = in.spe;
    // ...then the paralysis/burn stat drops are re-applied on switch-in (a Gen 1 volatile).
    if (in.status == Status::Paralysis) modify_stat(in, 3, 0.25);
    if (in.status == Status::Burn) modify_stat(in, 0, 0.5);
    // Note: Gen 1 does NOT tick burn/poison on switch-in (the conditions' onAfterSwitchInSelf
    // isn't triggered by gen1's engine — verified by differential testing).
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
    if (s.mon().must_recharge) {  // Hyper Beam recharge: locked to a single no-op move, no switch
        out.push_back({ChoiceKind::Move, -2});
        return out;
    }
    int avail = 0;
    for (int i = 0; i < static_cast<int>(s.mon().moves.size()); ++i)
        if (s.mon().pp[i] != 0) { out.push_back({ChoiceKind::Move, i}); ++avail; }  // pp 0 = exhausted
    if (avail == 0) out.push_back({ChoiceKind::Move, -1});  // all moves out of PP -> Struggle
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

    // Record the move each side SELECTED this turn (for Counter's desync clause): a real slot
    // (index >= 0) or Struggle (-1); recharge (-2) and switches select no move.
    auto selected = [&](const Side& s, const Choice& c, bool m) -> const MoveData* {
        if (m && c.index >= 0) return s.mon().moves[c.index];
        if (m && c.index == -1) return &STRUGGLE;
        return nullptr;
    };
    if (const MoveData* sm = selected(p1, c1, m1)) p1.last_selected = sm;
    if (const MoveData* sm = selected(p2, c2, m2)) p2.last_selected = sm;

    int s1 = effective_speed(p1.mon()), s2 = effective_speed(p2.mon());
    // Priority bracket (Counter = -5, moves last). Different brackets resolve deterministically.
    int pr1 = (m1 && c1.index >= 0) ? p1.mon().moves[c1.index]->priority : 0;
    int pr2 = (m2 && c2.index >= 0) ? p2.mon().moves[c2.index]->priority : 0;

    // Speed ties: Showdown burns extra random(0,2) "speed-tie shuffles" — but only among actions
    // in the SAME priority bracket with equal speed. 3 shuffles before the moves (the first
    // decides order), 1 between, 2 after; only the first changes observable state. Differing
    // priority (or distinct speed) consumes nothing, matching Showdown.
    bool tie = (m1 && m2 && pr1 == pr2 && s1 == s2);
    bool p1_first;
    if (pr1 != pr2) {
        p1_first = pr1 > pr2;  // higher priority first
    } else if (s1 != s2) {
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

    // Burn/poison ticks 1/16 right after the afflicted mon's own move (onAfterMoveSelf),
    // including a turn it's fully paralyzed/asleep (the move action still resolves) — but NOT
    // on a turn its move faints the target (Gen 1: AfterMoveSelf needs target.hp > 0).
    auto act1 = [&]() { if (m1) { try_move(p1, p2, c1.index, rng, last_damage); if (!p2.mon().fainted()) residual(p1.mon()); } };
    auto act2 = [&]() { if (m2) { try_move(p2, p1, c2.index, rng, last_damage); if (!p1.mon().fainted()) residual(p2.mon()); } };
    // If a side has no Pokémon left after the first move (Self-Destruct, Struggle recoil, or a
    // residual KO), the battle is over: Showdown stops the turn — no second move, no shuffles.
    if (p1_first) {
        act1();
        if (result() == Result::Ongoing) { if (tie) rng.random(0, 2); act2(); }
    } else {
        act2();
        if (result() == Result::Ongoing) { if (tie) rng.random(0, 2); act1(); }
    }
    if (result() == Result::Ongoing && tie) { rng.random(0, 2); rng.random(0, 2); }

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
