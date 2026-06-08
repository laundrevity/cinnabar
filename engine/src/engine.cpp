#include "cinnabar/engine.hpp"

#include "cinnabar/gen1_data.hpp"  // generated from Showdown: GEN1_TYPE_CHART, GEN1_SPECIES

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

// Set CINNABAR_RNG=1 to dump every PRNG draw to stderr (for diffing the RNG stream vs Showdown).
static const bool RNG_DEBUG = std::getenv("CINNABAR_RNG") != nullptr;

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
                                       e.priority, e.skip_lastdamage, e.drain_num, e.drain_den,
                                       e.needs_sleep_target, e.multihit_min, e.multihit_max});
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
    if (RNG_DEBUG) std::fprintf(stderr, "  ourrng random(%d) = %d\n", n, r);
    return r;
}
int RNG::random(int from, int to) {
    int r = static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(to - from)) >> 32) + from;
    if (RNG_DEBUG) std::fprintf(stderr, "  ourrng random(%d,%d) = %d\n", from, to, r);
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
            if (p.sleep_turns <= 0) { p.status = Status::None; p.status_by_foe = false; }  // wake: clears the clause hold
            return false;  // Gen 1: a turn is always lost to sleep, including the wake turn
        default:
            break;
    }
    if (p.partial_trapped > 0) return false;  // partially trapped (Wrap): lose the turn (priority 9)
    if (p.must_recharge) {        // Hyper Beam recharge: spend this turn doing nothing, clear the
        p.must_recharge = false;  // flag, and skip the paralysis roll (priority 7 > par's 2).
        return false;
    }
    // Disable (onBeforeMove priority 6, after recharge, before confusion): the timer ticks every turn
    // this mon gets to move. The disabled slot is already excluded from choices(), so this only counts
    // down (and frees the slot at 0) — it never cancels the move here.
    if (p.disable_turns > 0 && --p.disable_turns == 0) p.disable_slot = -1;
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

// Gen 1 recovery bug: Recover / Soft-Boiled / Rest fail at full HP, and when current HP is exactly
// 255 or 511 below max (i.e. (maxhp - hp + 1) is divisible by 256) unless HP is itself a multiple
// of 256. Showdown checks the two exact values (HP is always < 768, so only those are reachable).
bool recovery_fails(const Pokemon& p) {
    if (p.hp == p.max_hp) return true;
    int miss = p.max_hp - p.hp;
    return (miss == 255 || miss == 511) && p.hp % 256 != 0;
}

// Sleep / Freeze Clause (OU, training only): a fresh foe-inflicted `st` fails if another living
// mon on the target's side already carries that status from a foe's move. Off unless clauses are
// enabled, so the clause-free fidelity harness is unaffected. Rest-induced sleep never sets
// status_by_foe, so it's correctly exempt.
static bool clause_blocks(const Side& side, const Pokemon& tgt, Status st) {
    if (!side.clauses) return false;
    for (const auto& m : side.team)
        if (&m != &tgt && !m.fainted() && m.status == st && m.status_by_foe) return true;
    return false;
}

void apply_effect(const MoveData* mv, Pokemon& user, Pokemon& tgt, Side& tgt_side, RNG& rng) {
    if (mv->effect == Effect::None) return;
    switch (mv->effect) {
        case Effect::Heal:
            if (!recovery_fails(user)) user.hp = std::min(user.max_hp, user.hp + user.max_hp / 2);
            return;
        case Effect::Rest:
            if (recovery_fails(user)) return;  // same HP-mod-256 bug gates Rest (no sleep, no heal)
            user.status = Status::Sleep;
            user.sleep_turns = 2;
            user.hp = user.max_hp;
            return;
        case Effect::Reflect:
            user.reflect = true;
            return;
        case Effect::LightScreen:
            user.light_screen = true;
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
        case Effect::SuperFang:
        case Effect::Psywave:
            return;  // set-damage moves: damage handled in use_move, no secondary effect or RNG
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
    if (mv->effect == Effect::LeechSeed) {  // a volatile; re-seeding an already-seeded target just fails
        tgt.leech_seeded = true;
        return;
    }
    if (mv->effect == Effect::Disable) {  // bypasses Substitute (gen1 bypasssub flag)
        if (tgt.disable_turns > 0) return;  // already disabled -> addVolatile fails (accuracy already spent)
        // Sample a random move slot with PP > 0 (Showdown's this.sample over the filtered slots), then
        // roll the duration random(1,9) = 1-8 turns. These two draws happen only on a fresh disable.
        std::vector<int> pp_slots;
        for (size_t i = 0; i < tgt.pp.size(); ++i) if (tgt.pp[i] != 0) pp_slots.push_back(static_cast<int>(i));
        if (pp_slots.empty()) return;
        tgt.disable_slot = pp_slots[rng.random(static_cast<int>(pp_slots.size()))];
        tgt.disable_turns = rng.random(1, 9);
        return;
    }

    const bool secondary = mv->effect_chance < 100;  // <100 == a damaging move's secondary
    // Substitute (Gen 1): while the target's sub is up, a damaging move's secondary status is
    // skipped entirely (no RNG roll); a primary status move is blocked only for poison —
    // paralysis / sleep / freeze / burn pass through the substitute.
    if (tgt.has_substitute) {
        if (secondary) return;
        if (mv->effect == Effect::Poison || mv->effect == Effect::Toxic) return;
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
        case Effect::Sleep:
            if (clause_blocks(tgt_side, tgt, Status::Sleep)) break;  // Sleep Clause: 2nd foe-sleep fails
            tgt.status = Status::Sleep; tgt.sleep_turns = rng.range(1, 7); tgt.status_by_foe = true;
            break;
        case Effect::Freeze:
            if (clause_blocks(tgt_side, tgt, Status::Freeze)) break;  // Freeze Clause: 2nd foe-freeze fails
            tgt.status = Status::Freeze; tgt.status_by_foe = true;
            break;
        case Effect::Burn:     tgt.status = Status::Burn; modify_stat(tgt, 0, 0.5); break;
        case Effect::Poison:   tgt.status = Status::Poison; break;
        case Effect::Toxic:    tgt.status = Status::Poison; tgt.toxic = true; tgt.tox_stage = 0; break;
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
        // Deduct PP on use (gen1: even if it misses/fails) — EXCEPT on a partial-trap continuation:
        // a semi-locked move (Wrap/Clamp/Fire Spin/Bind) costs PP only on its initiating turn, not
        // on the auto-repeated locked turns (Showdown: "Locked moves don't deduct PP").
        const bool trap_continue = (mv->effect == Effect::Trap && a.wrap_turns > 0);
        if (!trap_continue && a.pp[moveidx] > 0) --a.pp[moveidx];
    }
    as.last_move = mv;  // record the move used (Counter reads the opponent's last used move)
    if (!mv->skip_lastdamage) last_damage = 0;  // damaging non-Counter moves clear it; Counter/status don't
    int recoil_base = 0;     // damage the recoil is computed from (capped; uncapped vs a sub)
    bool recoil_ok = false;  // recoil applies (false vs a sub that broke on this hit)

    // Moves that target the user (Recover/Rest/Reflect/Substitute, self-boosts) skip the
    // accuracy roll and the immunity check below.
    bool self_targeting = mv->effect == Effect::Heal || mv->effect == Effect::Rest ||
                          mv->effect == Effect::Reflect || mv->effect == Effect::LightScreen ||
                          mv->effect == Effect::Substitute ||
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
    // Dream Eater fails (at the immunity step, before the accuracy roll, so no RNG) unless the
    // target is asleep — Showdown gates it with onTryImmunity(target.status === 'slp').
    if (mv->needs_sleep_target && d.status != Status::Sleep) return;
    // Leech Seed fails on Grass-type targets (onTryImmunity), again before the accuracy roll.
    if (mv->effect == Effect::LeechSeed &&
        (d.species->t1 == Type::Grass || d.species->t2 == Type::Grass)) return;
    // Poison types can't be poisoned: a primary poison move (Toxic / Poison Powder) fails before
    // accuracy. (Damaging moves with a poison *secondary* are handled in the secondary roll path.)
    if ((mv->effect == Effect::Toxic || (mv->effect == Effect::Poison && mv->effect_chance >= 100)) &&
        (d.species->t1 == Type::Poison || d.species->t2 == Type::Poison)) return;
    // Disable's onTryHit fails (before accuracy, no RNG) if the target has no move with PP left.
    if (mv->effect == Effect::Disable) {
        bool any_pp = false;
        for (size_t i = 0; i < d.pp.size(); ++i) if (d.pp[i] != 0) { any_pp = true; break; }
        if (!any_pp) return;
    }

    // Accuracy — Showdown gen1 rolls randomChance(clamp(floor(acc*255/100),1,255), 256) for
    // every non-self-targeting move, *including* 100%-accuracy ones (the 1/256 miss). A continuing
    // partial-trap (Wrap) auto-hits its locked target, so it skips the roll after the first turn.
    bool wrap_continue = (mv->effect == Effect::Trap && a.wrap_turns > 0);
    if (!self_targeting && !wrap_continue && mv->accuracy > 0) {
        int acc = std::clamp(mv->accuracy * 255 / 100, 1, 255);
        if (!rng.chance(acc, 256)) {  // miss (includes the gen1 1/256)
            if (mv->effect == Effect::SelfDestruct) a.hp = 0;  // gen1: Explosion faints user on a miss
            last_damage = 0;  // a miss clears last_damage (Counter can't reflect a missed hit)
            return;
        }
    }

    // Multi-hit count (Pin Missile / Double Kick / ...). Sampled AFTER accuracy and BEFORE the per-hit
    // damage rolls (gen1 order). The [2,5] moves use the same fixed distribution as Wrap's duration;
    // a fixed count (Double Kick = 2) consumes no RNG. In gen1 every hit deals the FIRST hit's damage
    // (getDamage returns the stored value for hit>1), so crit/range are rolled once below and reused.
    int multihits = 1;
    if (mv->multihit_max > 0) {
        if (mv->multihit_min == 2 && mv->multihit_max == 5) {
            static const int MH[8] = {2, 2, 2, 3, 3, 3, 4, 5};
            multihits = MH[rng.random(8)];
        } else if (mv->multihit_min == mv->multihit_max) {
            multihits = mv->multihit_min;  // fixed count, no RNG
        } else {
            multihits = rng.random(mv->multihit_min, mv->multihit_max + 1);
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

    if (mv->effect == Effect::Trap && wrap_continue) {
        // Continuing partial-trap: re-deal the STORED first-turn damage — Gen 1 partial-trap damage
        // is fixed, so no crit/damage/accuracy rolls happen on continuation (keeps RNG aligned).
        int dmg = a.wrap_damage;
        if (d.has_substitute) {
            d.sub_hp -= dmg > d.sub_hp ? d.sub_hp : dmg;
            if (d.sub_hp <= 0) { d.has_substitute = false; d.sub_hp = 0; }
            last_damage = dmg;
        } else {
            int hit = dmg < d.hp ? dmg : d.hp;
            d.hp -= hit;
            last_damage = hit;
        }
    } else if (mv->effect == Effect::SuperFang || mv->effect == Effect::Psywave) {
        // Set-damage moves whose amount comes from a Gen 1 damageCallback (not the type/STAB formula):
        //   Super Fang = max(1, floor(target.hp / 2))          — no RNG
        //   Psywave    = random(0, floor(1.5 * level))         — one RNG draw (0..149 at L100); a 0
        //                roll makes the move fail (0 damage), which falls out naturally below.
        // Applied like fixed damage: uncapped vs a Substitute, capped at HP vs a bare target.
        int dmg = (mv->effect == Effect::SuperFang) ? std::max(1, d.hp / 2)
                                                    : rng.random(0, (3 * a.level) / 2);
        if (d.has_substitute) {
            d.sub_hp -= dmg > d.sub_hp ? d.sub_hp : dmg;
            if (d.sub_hp <= 0) { d.has_substitute = false; d.sub_hp = 0; }
            last_damage = dmg;
        } else {
            int hit = dmg < d.hp ? dmg : d.hp;
            d.hp -= hit;
            last_damage = hit;
        }
    } else if (mv->fixed > 0) {
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
            if (physical && d.reflect) def *= 2;        // Reflect doubles Def (a screen, not on crit)
            if (!physical && d.light_screen) def *= 2;  // Light Screen doubles Spc def (special side)
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
        // Apply the hit `multihits` times (1 for a normal move). Each hit deals the same `dmg`
        // (gen1 reuses hit 1's damage), capped per-hit at the target's current HP; the run stops
        // when the target faints or its Substitute breaks. last_damage = the LAST hit (Counter).
        for (int h = 0; h < multihits; ++h) {
            if (d.fainted()) break;
            if (d.has_substitute) {  // Gen 1: damage hits the sub; excess is NOT dealt to real HP
                d.sub_hp -= dmg > d.sub_hp ? d.sub_hp : dmg;
                recoil_ok = d.sub_hp > 0;  // recoil happens only if the sub survived (Gen 1)
                recoil_base = dmg;         // uncapped (the Gen 1 sub quirk)
                last_damage = dmg;
                if (d.sub_hp <= 0) { d.has_substitute = false; d.sub_hp = 0; break; }  // sub broke -> stop
            } else {
                int actual = dmg < d.hp ? dmg : d.hp;  // Showdown caps damage at the target's HP
                d.hp -= actual;
                recoil_base = actual;
                recoil_ok = true;
                last_damage = actual;  // what a subsequent Counter would double
            }
        }
    }

    if (mv->effect == Effect::SelfDestruct) a.hp = 0;
    // Recoil to the user (Double-Edge/Take Down/Submission/Struggle): floor(dmg * num/den), min 1.
    // Vs a Substitute, Gen 1 deals recoil only if the sub survived, off the uncapped damage.
    if (mv->recoil_den > 0 && recoil_ok && recoil_base > 0) {
        a.hp -= std::max(1, recoil_base * mv->recoil_num / mv->recoil_den);
        if (a.hp < 0) a.hp = 0;
    }
    // Drain (Mega Drain / Absorb / Leech Life / Dream Eater): heal the user floor(damage * num/den),
    // min 1, off the SAME damage figure recoil uses — capped vs a bare target, uncapped vs a
    // Substitute, and (like recoil) only if the Substitute survived. The heal consumes no RNG.
    if (mv->drain_den > 0 && recoil_ok && recoil_base > 0) {
        a.hp += std::max(1, recoil_base * mv->drain_num / mv->drain_den);
        if (a.hp > a.max_hp) a.hp = a.max_hp;
    }
    apply_effect(mv, a, d, ds, rng);
    apply_boosts(mv, a, d, rng);
    // Hyper Beam: the user owes a recharge turn — UNLESS the move KO'd the target (the famous
    // Gen 1 "no recharge on KO"). Reaching here means the move hit (a miss/immunity returned early).
    if (mv->recharge && !d.fainted()) a.must_recharge = true;
    if (mv->effect == Effect::Trap) {
        // Partial-trap (Wrap/Bind/Fire Spin/Clamp): lock the user re-using this move for a sampled
        // 2-5 turns and hold the foe (it loses its turn). Durations tick at END of turn (step()),
        // so the foe stays trapped through the final turn. On a non-final turn Showdown's onAfterMove
        // re-adds partiallytrapped to the foe; on the FINAL turn (and only if the lock wasn't the
        // max 5-turn roll) it instead adds the cosmetic `fakepartiallytrapped` to BOTH mons for two
        // turns. That volatile has no battle effect, but it carries a `duration`, so its handler joins
        // the end-of-turn Residual fieldEvent's speed-sort and consumes shuffle RNG (residual_trap_shuffle).
        if (!wrap_continue) {  // first hit: sample the lock duration and store the fixed damage
            static const int DUR[8] = {2, 2, 2, 3, 3, 3, 4, 5};
            a.wrap_turns = DUR[rng.random(8)];  // sampled even on a KO (onStart runs before removal)
            a.wrap_total = a.wrap_turns;
            a.wrap_idx = moveidx;
            a.wrap_damage = recoil_base;
        }
        if (d.fainted()) {               // a KO removes the lock (Showdown's onAfterMove early-returns)
            a.wrap_turns = 0; a.wrap_idx = -1;
        } else if (a.wrap_turns != 1) {  // not the final turn -> keep the foe held
            d.partial_trapped = 2;
        } else if (a.wrap_total != 5) {  // final turn -> fakepartiallytrapped on both (unless 5-turn lock)
            a.fake_trap_turns = 2;
            d.fake_trap_turns = 2;
        }
    }
}

void try_move(Side& as, Side& ds, int moveidx, RNG& rng, int& last_damage) {
    if (as.mon().fainted()) return;
    // If the foe already fainted earlier this turn (e.g. it KO'd itself on Struggle/recoil before
    // this — the slower — mon moved), Gen 1 cancels this move entirely: no accuracy roll, no RNG, no
    // can_act check. The fainted side switches at end of turn, so the move never resolves.
    if (ds.mon().fainted()) return;
    if (!can_act(as.mon(), rng)) return;  // sleep/freeze/recharge/full-para/confusion self-hit
    use_move(as, ds, moveidx, rng, last_damage);
}

void residual(Pokemon& p) {
    if (p.fainted()) return;
    if (p.status == Status::Poison && p.toxic) {
        // Badly poisoned: damage = stage * floor(maxhp/16), with the stage incrementing (cap 15)
        // before each tick. So the first tick is 1x, then 2x, 3x, ... (resets to 0 on switch).
        if (p.tox_stage < 15) ++p.tox_stage;
        p.hp -= std::max(1, p.max_hp / 16) * p.tox_stage;
        if (p.hp < 0) p.hp = 0;
    } else if (p.status == Status::Burn || p.status == Status::Poison) {
        p.hp -= std::max(1, p.max_hp / 16);
        if (p.hp < 0) p.hp = 0;
    }
}

void do_switch(Side& s, int idx) {
    s.mon().reflect = false;       // Reflect ends when its user leaves the field
    s.mon().light_screen = false;  // Light Screen likewise clears on switch out
    s.mon().leech_seeded = false;  // Leech Seed is shed on switch out
    s.mon().tox_stage = 0;         // Gen 1: the toxic counter resets on switch (status 'tox' persists)
    s.mon().disable_slot = -1; s.mon().disable_turns = 0;  // Disable clears when its target leaves
    s.active = idx;
    Pokemon& in = s.mon();
    in.must_recharge = false;  // volatiles clear on switch (a recharge can't carry to a new mon)
    in.has_substitute = false; in.sub_hp = 0;  // a Substitute does not persist across a switch
    in.confuse_turns = 0;      // confusion clears on switch out
    in.wrap_turns = 0; in.wrap_total = 0; in.wrap_idx = -1; in.wrap_damage = 0;  // trap clears on switch
    in.partial_trapped = 0; in.fake_trap_turns = 0;
    // Gen 1: stat stages reset on switch — recompute modified stats from the stored stats...
    in.boost_atk = in.boost_def = in.boost_spc = in.boost_spe = 0;
    in.m_atk = in.atk; in.m_def = in.def; in.m_spc = in.spc; in.m_spe = in.spe;
    // ...then the paralysis/burn stat drops are re-applied on switch-in (a Gen 1 volatile).
    if (in.status == Status::Paralysis) modify_stat(in, 3, 0.25);
    if (in.status == Status::Burn) modify_stat(in, 0, 0.5);
    // Note: Gen 1 does NOT tick burn/poison on switch-in (the conditions' onAfterSwitchInSelf
    // isn't triggered by gen1's engine — verified by differential testing).
}

// End-of-turn Residual fieldEvent shuffle. Showdown's fieldEvent('Residual') collects every handler
// whose state carries a `duration`, speed-sorts them (Battle.speedSort), and PRNG-shuffles each
// maximal same-speed group. In Gen 1 the ONLY duration-bearing states are the three partial-trap
// volatiles — sleep/freeze/confusion track `time`, not `duration`, so they never enter this sort
// (which is why non-wrap battles consume no shuffle RNG). We don't model handler objects, so we just
// reproduce the shuffle's RNG draws to keep the stream aligned (the sort result itself is cosmetic):
//   - partialtrappinglock  -> on the wrapper while locked        (wrap_turns > 0)
//   - partiallytrapped     -> on the victim while held           (partial_trapped > 0)
//   - fakepartiallytrapped -> on BOTH for two turns at a boundary (fake_trap_turns > 0)
// Handlers tie iff they share a holder speed, so each mon's handlers form one same-speed group.
void residual_trap_shuffle(Side& p1, Side& p2, RNG& rng) {
    auto handler_count = [](const Pokemon& m) {
        if (m.fainted()) return 0;  // fieldEvent skips handlers whose holder has fainted
        return (m.wrap_turns > 0 ? 1 : 0) + (m.partial_trapped > 0 ? 1 : 0) + (m.fake_trap_turns > 0 ? 1 : 0);
    };
    int h1 = handler_count(p1.mon()), h2 = handler_count(p2.mon());
    if (h1 + h2 < 2) return;  // speedSort is a no-op below two handlers
    int s1 = effective_speed(p1.mon()), s2 = effective_speed(p2.mon());
    // prng.shuffle(list, start, end): for i in [start, end-1) draw random(i, end). A group of size k
    // therefore burns k-1 draws. speedSort processes groups fastest-first; equal speed merges them.
    auto shuffle_group = [&](int start, int end) {
        for (int i = start; i + 1 < end; ++i) rng.random(i, end);
    };
    if (s1 == s2) {
        shuffle_group(0, h1 + h2);  // both holders share a speed -> a single tied group
    } else {
        int hf = (s1 > s2) ? h1 : h2;  // faster holder's handlers occupy the front of the list
        int hs = (s1 > s2) ? h2 : h1;
        if (hf >= 2) shuffle_group(0, hf);
        if (hs >= 2) shuffle_group(hf, hf + hs);
    }
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
    if (s.mon().wrap_turns > 0) {  // partial-trap: locked re-using the move, no switch
        out.push_back({ChoiceKind::Move, s.mon().wrap_idx});
        return out;
    }
    int avail = 0;
    for (int i = 0; i < static_cast<int>(s.mon().moves.size()); ++i) {
        if (s.mon().pp[i] == 0) continue;                                      // pp 0 = exhausted
        if (s.mon().disable_turns > 0 && i == s.mon().disable_slot) continue;  // Disable: slot unselectable
        out.push_back({ChoiceKind::Move, i});
        ++avail;
    }
    if (avail == 0) out.push_back({ChoiceKind::Move, -1});  // all moves out of PP / disabled -> Struggle
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

    // Leech Seed drains the seeded mon for 1/16 max HP right after ITS OWN move (gen1 onAfterMoveSelf,
    // priority 1 — after burn/poison's priority 2), healing the leecher. Gen 1 recovery is NOT limited
    // by the seeded mon's remaining HP (the leecher gets the full amount). No RNG.
    auto leech = [](Pokemon& seeded, Pokemon& leecher) {
        if (!seeded.leech_seeded || leecher.fainted()) return;
        int toLeech = std::max(1, seeded.max_hp / 16);
        if (seeded.hp > 0) seeded.hp -= toLeech < seeded.hp ? toLeech : seeded.hp;  // drain (capped)
        // Gen 1: the leecher recovers the full amount even if the seeded mon had less HP — or already
        // fainted to its OWN move's recoil this turn (the leech still fires and heals the leecher).
        leecher.hp += toLeech;
        if (leecher.hp > leecher.max_hp) leecher.hp = leecher.max_hp;
    };
    // Burn/poison ticks 1/16 right after the afflicted mon's own move (onAfterMoveSelf),
    // including a turn it's fully paralyzed/asleep (the move action still resolves) — but NOT
    // on a turn its move faints the target (Gen 1: AfterMoveSelf needs target.hp > 0). Leech Seed
    // shares that gate (the whole AfterMoveSelf event is skipped when the move KO'd its target).
    auto act1 = [&]() { if (m1) { try_move(p1, p2, c1.index, rng, last_damage); if (!p2.mon().fainted()) { residual(p1.mon()); leech(p1.mon(), p2.mon()); } } };
    auto act2 = [&]() { if (m2) { try_move(p2, p1, c2.index, rng, last_damage); if (!p1.mon().fainted()) { residual(p2.mon()); leech(p2.mon(), p1.mon()); } } };
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

    // End-of-turn Residual fieldEvent: speed-sort + shuffle the duration-bearing (partial-trap)
    // handlers. Runs before the duration tick (Showdown sorts, then decrements in sorted order).
    if (result() == Result::Ongoing) residual_trap_shuffle(p1, p2, rng);

    // Partial-trap durations tick down at end of turn (Showdown's duration system), so the foe
    // stays held through the wrapper's final turn and is freed the turn after.
    auto tick = [](Pokemon& m) {
        if (m.wrap_turns > 0 && --m.wrap_turns == 0) m.wrap_idx = -1;
        if (m.partial_trapped > 0) --m.partial_trapped;
        if (m.fake_trap_turns > 0) --m.fake_trap_turns;
    };
    tick(p1.mon());
    tick(p2.mon());

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
