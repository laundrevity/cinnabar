// See encoder.hpp. Every block below is a transcription of the Python encoder
// (encoding.py / engine_cpp.py build_state) or of the heuristic pilots (policy.py);
// comments cite the Python it mirrors. Quirks are preserved deliberately — parity with
// the trained nets' observation distribution matters more than tidiness here.
#include "cinnabar/encoder.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cinnabar::enc {

namespace {

double g_chart[kTypes][kTypes];  // [defender][attacker], like StaticData._chart
bool g_chart_set = false;
std::unordered_map<std::string, MoveMeta> g_moves;      // keyed by lowercase-alnum id
std::unordered_map<std::string, SpeciesMeta> g_species;
// Name-normalization happens once per distinct MoveData*/Species* (they're static table
// entries), not per encode call — the string churn was exactly the Python bottleneck.
std::unordered_map<const MoveData*, const MoveMeta*> g_move_cache;
std::unordered_map<const Species*, const SpeciesMeta*> g_species_cache;
const MoveMeta g_default_move{};        // == StaticData.move_meta on an unknown id ("Recharge")
const SpeciesMeta g_default_species{};

std::string to_id(const std::string& name) {  // engine_cpp._to_id
    std::string out;
    out.reserve(name.size());
    for (char c : name)
        if (std::isalnum(static_cast<unsigned char>(c)))
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

const MoveMeta& meta(const MoveData* md) {
    auto it = g_move_cache.find(md);
    if (it != g_move_cache.end()) return *it->second;
    auto reg = g_moves.find(to_id(md->name));
    const MoveMeta* m = (reg != g_moves.end()) ? &reg->second : &g_default_move;
    g_move_cache.emplace(md, m);
    return *m;
}

const MoveMeta& meta_by_id(const std::string& id) {
    auto reg = g_moves.find(id);
    return (reg != g_moves.end()) ? reg->second : g_default_move;
}

const SpeciesMeta& meta(const Species* sp) {
    auto it = g_species_cache.find(sp);
    if (it != g_species_cache.end()) return *it->second;
    auto reg = g_species.find(to_id(sp->name));
    const SpeciesMeta* m = (reg != g_species.end()) ? &reg->second : &g_default_species;
    g_species_cache.emplace(sp, m);
    return *m;
}

// StaticData.type_mult: product of chart[defender][attacker] over the defender's types.
double type_mult(int atk_idx, const SpeciesMeta& def) {
    double m = 1.0;
    for (int8_t t : def.types)
        if (t >= 0) m *= g_chart[t][atk_idx];
    return m;
}

bool has_type(const SpeciesMeta& sp, int idx) {
    return sp.types[0] == idx || sp.types[1] == idx;
}

// encoding._status_one_hot index (STATUS_ORDER: NONE BRN FRZ PAR PSN SLP; TOX folds to PSN,
// which the engine already does — Status::Poison covers toxic).
int status_idx(const Pokemon& m) {
    switch (m.status) {
        case Status::Burn: return 1;
        case Status::Freeze: return 2;
        case Status::Paralysis: return 3;
        case Status::Poison: return 4;
        case Status::Sleep: return 5;
        default: return 0;
    }
}

// encoding._effective_speed: BASE species speed, quartered (floor) when paralyzed; -1 unknown.
int effective_speed(const Pokemon& m, const SpeciesMeta& sp) {
    if (sp.speed < 0) return -1;
    return m.status == Status::Paralysis ? sp.speed / 4 : sp.speed;
}

// Per-choice facts shared by encode() and the heuristic pilots — the same quantities
// build_state puts on each Action.
struct Facts {
    bool is_move = false;
    const MoveMeta* mm = nullptr;  // moves only
    double mult = 1.0;             // move: type effectiveness vs the opponent's active
    bool stab = false;
    double tgt_hp = 0.0;           // switches only: what we'd switch into
    bool tgt_statused = false;
    double incoming = 0.0;         // max over opp-active types vs the switch target
};

void build_facts(const Battle& b, int player, std::vector<Choice>& choices,
                 std::vector<Facts>& out) {
    const Side& me = (player == 0) ? b.p1 : b.p2;
    const Side& op = (player == 0) ? b.p2 : b.p1;
    const SpeciesMeta& my_sp = meta(me.mon().species);
    const SpeciesMeta& op_sp = meta(op.mon().species);
    choices = b.choices(player);
    out.clear();
    out.reserve(choices.size());
    for (const Choice& c : choices) {
        Facts f;
        if (c.kind == ChoiceKind::Move) {
            f.is_move = true;
            // build_state: -1 -> "Struggle" (a real poke-env entry), -2 -> "Recharge"
            // (unknown id -> move_meta defaults).
            f.mm = (c.index == -1) ? &meta_by_id("struggle")
                 : (c.index == -2) ? &g_default_move
                                   : &meta(me.mon().moves[c.index]);
            f.mult = type_mult(f.mm->type_idx, op_sp);
            f.stab = has_type(my_sp, f.mm->type_idx);
        } else {  // Switch (choices() never emits Pass)
            const Pokemon& tgt = me.team[c.index];
            const SpeciesMeta& tgt_sp = meta(tgt.species);
            f.tgt_hp = tgt.hp_fraction();
            f.tgt_statused = tgt.status != Status::None;
            double inc = 0.0;  // max(...) over the opponent's types, like build_state
            for (int8_t t : op_sp.types)
                if (t >= 0) inc = std::max(inc, type_mult(t, tgt_sp));
            f.incoming = inc;
        }
        out.push_back(f);
    }
}

bool revealed(const Observer* obs, int slot) {
    return obs == nullptr || (obs->mons >> slot & 1);
}

// CPython's builtin sum() compensates float addition (Neumaier) since 3.12; the HP sums
// feed both a feature and the material bookkeeping, so mirror it for bit parity.
struct NeumaierSum {
    double total = 0.0, c = 0.0;
    void add(double x) {
        double t = total + x;
        if (std::abs(total) >= std::abs(x))
            c += (total - t) + x;
        else
            c += (x - t) + total;
        total = t;
    }
    double value() const { return total + c; }
};

}  // namespace

void register_type_chart(const double chart_def_atk[kTypes][kTypes]) {
    for (int d = 0; d < kTypes; ++d)
        for (int a = 0; a < kTypes; ++a) g_chart[d][a] = chart_def_atk[d][a];
    g_chart_set = true;
}

void register_move(const std::string& id, const MoveMeta& m) {
    g_moves[id] = m;
    g_move_cache.clear();  // value addresses may move on rehash
}

void register_species(const std::string& id, const SpeciesMeta& s) {
    g_species[id] = s;
    g_species_cache.clear();
}

bool encoder_ready() { return g_chart_set && !g_moves.empty() && !g_species.empty(); }

int encode(const Battle& b, int player, Observer* obs, float* g, float* a,
           double* my_mat, double* opp_mat) {
    if (!encoder_ready()) throw std::runtime_error("encoder not registered (call register_encoder)");
    const Side& me = (player == 0) ? b.p1 : b.p2;
    const Side& op = (player == 0) ? b.p2 : b.p1;
    const Pokemon& my = me.mon();
    const Pokemon& opp = op.mon();
    const SpeciesMeta& my_sp = meta(my.species);
    const SpeciesMeta& op_sp = meta(opp.species);

    if (obs) obs->mons |= static_cast<uint8_t>(1u << op.active);  // build_state: reveal their active

    // ---- global vector (encoding.encode_global, field for field) ------------------------
    int gi = 0;
    auto put = [&](double v) { g[gi++] = static_cast<float>(v); };

    put(my.hp_fraction());
    put(opp.hp_fraction());
    for (int i = 0; i < kStatusOneHot; ++i) put(i == status_idx(my) ? 1.0 : 0.0);
    for (int i = 0; i < kStatusOneHot; ++i) put(i == status_idx(opp) ? 1.0 : 0.0);
    put(me.must_switch ? 1.0 : 0.0);
    put(std::min(b.turn / 50.0, 1.0));  // _TURN_SCALE

    int our_alive = 0;
    NeumaierSum our_hp_total;
    for (const Pokemon& m : me.team) {
        if (!m.fainted()) ++our_alive;
        our_hp_total.add(m.hp_fraction());
    }
    int opp_revealed_n = 0, opp_fainted = 0;
    NeumaierSum opp_hp_total;
    bool opp_any_slp = false, opp_any_frz = false;
    for (int i = 0; i < static_cast<int>(op.team.size()); ++i) {
        if (!revealed(obs, i)) continue;
        ++opp_revealed_n;
        const Pokemon& m = op.team[i];
        if (m.fainted()) ++opp_fainted;
        opp_hp_total.add(m.hp_fraction());
        opp_any_slp |= m.status == Status::Sleep;   // clause features scan REVEALED mons only,
        opp_any_frz |= m.status == Status::Freeze;  // exactly like state.opponent_team
    }
    put(our_alive / static_cast<double>(kTeam));
    put(our_hp_total.value() / kTeam);
    put(opp_fainted / static_cast<double>(kTeam));
    put(opp_revealed_n / static_cast<double>(kTeam));

    int s_me = effective_speed(my, my_sp), s_op = effective_speed(opp, op_sp);
    put((s_me < 0 || s_op < 0 || s_me == s_op) ? 0.5 : (s_me > s_op ? 1.0 : 0.0));

    put(my.boost_atk / 6.0);
    put(my.boost_def / 6.0);
    put(my.boost_spc / 6.0);
    put(my.boost_spe / 6.0);
    put(opp.boost_atk / 6.0);
    put(opp.boost_def / 6.0);
    put(opp.boost_spc / 6.0);
    put(opp.boost_spe / 6.0);
    put(my.must_recharge ? 1.0 : 0.0);
    put(opp.must_recharge ? 1.0 : 0.0);
    put((my.status == Status::Sleep ? my.sleep_turns : 0) / 7.0);
    put((opp.status == Status::Sleep ? opp.sleep_turns : 0) / 7.0);

    // Revealed-opponent team block: the j-th revealed mon (team order) fills slot j —
    // a compacted list, not team positions (encoding._encode_opp_team over opponent_team).
    int slot = 0;
    for (int i = 0; i < static_cast<int>(op.team.size()) && slot < kTeam; ++i) {
        if (!revealed(obs, i)) continue;
        const Pokemon& m = op.team[i];
        const SpeciesMeta& sp = meta(m.species);
        put(1.0);
        put(m.hp_fraction());
        put(m.fainted() ? 1.0 : 0.0);
        put(m.status != Status::None ? 1.0 : 0.0);
        for (int t = 0; t < kTypes; ++t) put(has_type(sp, t) ? 1.0 : 0.0);
        ++slot;
    }
    for (; slot < kTeam; ++slot)
        for (int k = 0; k < kOppMonDim; ++k) put(0.0);

    // Threat profile of the active opponent's revealed moves (encoding._encode_opp_moves).
    // Python expands the revealed move-ID set back over the full moveset (a duplicated move
    // counts twice once either copy is seen) — reproduce that by collecting revealed
    // MoveData pointers, then re-scanning all slots for membership.
    {
        double prof[kOppMoveDim] = {0, 0, 0, 0, 0, 0, 0};
        if (obs) {
            uint8_t mask = obs->moves[op.active];
            const auto& mvs = opp.moves;
            std::vector<const MoveData*> seen;  // revealed move identities (<= 4 in Gen 1)
            for (int i = 0; i < static_cast<int>(mvs.size()); ++i)
                if (mask >> i & 1) seen.push_back(mvs[i]);
            if (!seen.empty()) {
                int cnt = 0;
                double maxpow = 0.0, maxmult = 0.0;
                bool slp = false, par = false, anystat = false, rech = false;
                for (const MoveData* md : mvs) {
                    if (std::find(seen.begin(), seen.end(), md) == seen.end()) continue;
                    const MoveMeta& mm = meta(md);
                    ++cnt;
                    maxpow = std::max(maxpow, mm.base_power);
                    maxmult = std::max(maxmult, type_mult(mm.type_idx, my_sp));
                    slp |= mm.effect_status == 0;
                    par |= mm.effect_status == 1;
                    anystat |= mm.effect_status >= 0 || mm.is_status;
                    rech |= mm.recharge;
                }
                prof[0] = std::min(cnt / 4.0, 1.0);
                prof[1] = maxpow / 200.0;  // _MAX_BASE_POWER
                prof[2] = std::min(maxmult / 4.0, 1.0);
                prof[3] = slp ? 1.0 : 0.0;
                prof[4] = par ? 1.0 : 0.0;
                prof[5] = anystat ? 1.0 : 0.0;
                prof[6] = rech ? 1.0 : 0.0;
            }
        }
        for (double v : prof) put(v);
    }

    auto put_vol = [&](const Pokemon& m) {  // encoding's _vol: 6 bools per side
        put(m.confuse_turns > 0 ? 1.0 : 0.0);
        put(m.reflect ? 1.0 : 0.0);
        put(m.light_screen ? 1.0 : 0.0);
        put(m.leech_seeded ? 1.0 : 0.0);
        put(m.disable_turns > 0 ? 1.0 : 0.0);
        put(m.toxic ? 1.0 : 0.0);
    };
    put_vol(my);
    put_vol(opp);
    put(opp_any_slp ? 1.0 : 0.0);  // clause state, appended last (stable indices)
    put(opp_any_frz ? 1.0 : 0.0);

    if (gi != kGlobalDim) throw std::logic_error("encoder global-dim mismatch");

    if (my_mat) *my_mat = our_hp_total.value();
    if (opp_mat) *opp_mat = opp_hp_total.value();
    if (a == nullptr)  // global-only encode (search leaves need no action features)
        return 0;

    // ---- per-action rows (encoding.encode_action) ----------------------------------------
    std::vector<Choice> choices;
    std::vector<Facts> facts;
    build_facts(b, player, choices, facts);
    bool opp_active_statused = opp.status != Status::None;
    for (size_t i = 0; i < facts.size(); ++i) {
        const Facts& f = facts[i];
        float* row = a + i * kActionDim;
        int ai = 0;
        auto putr = [&](double v) { row[ai++] = static_cast<float>(v); };
        if (f.is_move) {
            const MoveMeta& mm = *f.mm;
            bool will_fail =
                (mm.effect_status >= 0 && mm.is_status && mm.effect_chance >= 0.999
                 && opp_active_statused)               // (a) primary status into a statused target
                || (mm.effect_status == 0 && opp_any_slp)   // (b) Sleep Clause spent
                || (mm.effect_status == 2 && opp_any_frz);  // (b) Freeze Clause spent
            putr(1.0);
            putr(0.0);
            putr(mm.base_power / 200.0);
            putr(f.mult);
            putr(f.stab ? 1.0 : 0.0);
            putr(mm.is_status ? 1.0 : 0.0);
            putr(mm.accuracy);
            putr(0.0);  // target_hp
            putr(0.0);  // target_statused
            putr(0.0);  // incoming
            putr(mm.fixed / 100.0);
            for (int s = 0; s < 5; ++s) putr(mm.effect_status == s ? 1.0 : 0.0);
            putr(mm.effect_chance);
            putr(mm.heals ? 1.0 : 0.0);
            putr(mm.boosts_self ? 1.0 : 0.0);
            putr(mm.lowers_foe ? 1.0 : 0.0);
            putr(mm.recharge ? 1.0 : 0.0);
            putr(mm.self_destruct ? 1.0 : 0.0);
            putr(will_fail ? 1.0 : 0.0);
        } else {
            putr(0.0);
            putr(1.0);
            putr(0.0);
            putr(1.0);  // type_multiplier None -> 1.0 for switches
            putr(0.0);
            putr(0.0);
            putr(1.0);  // accuracy None -> 1.0
            putr(f.tgt_hp);
            putr(f.tgt_statused ? 1.0 : 0.0);
            putr(f.incoming);
            for (int k = 0; k < 13; ++k) putr(0.0);  // fixed + one-hot(5) + chance + flags(5) + will_fail
        }
        if (ai != kActionDim) throw std::logic_error("encoder action-dim mismatch");
    }
    return static_cast<int>(facts.size());
}

Result step_pair(Battle& b, int i1, int i2, Observer* obs1, Observer* obs2) {
    std::vector<Choice> c1 = b.choices(0), c2 = b.choices(1);
    const Choice& a1 = c1.at(static_cast<size_t>(i1));
    const Choice& a2 = c2.at(static_cast<size_t>(i2));
    // engine_cpp.reveal_move: each side remembers the move the OTHER side selected this
    // turn (Struggle/Recharge excluded); switches reveal via encode() next turn instead.
    if (obs1 && a2.kind == ChoiceKind::Move && a2.index >= 0)
        obs1->moves[b.p2.active] |= static_cast<uint8_t>(1u << a2.index);
    if (obs2 && a1.kind == ChoiceKind::Move && a1.index >= 0)
        obs2->moves[b.p1.active] |= static_cast<uint8_t>(1u << a1.index);
    return b.step(a1, a2);
}

// ---- heuristic pilots (policy.py, decision-for-decision) -----------------------------------

namespace {

// SmartHeuristicPolicy._dmg: fixed damage short-circuits (no accuracy weighting there).
double smart_dmg(const Facts& f) {
    if (f.mm->base_power == 0.0 && f.mm->fixed > 0.0) return f.mm->fixed;
    return f.mm->base_power * f.mult * (f.stab ? 1.5 : 1.0) * f.mm->accuracy;
}

// `(s.incoming_multiplier or 1.0)`: Python's `or` maps BOTH None and a true 0.0 to 1.0.
double incoming_or_1(const Facts& f) { return f.incoming == 0.0 ? 1.0 : f.incoming; }

// _best_switch: first-minimal by (incoming-or-1, -target_hp). -1 if no switches.
int best_switch(const std::vector<Facts>& facts) {
    int best = -1;
    double bk1 = 0.0, bk2 = 0.0;
    for (size_t i = 0; i < facts.size(); ++i) {
        if (facts[i].is_move) continue;
        double k1 = incoming_or_1(facts[i]), k2 = -facts[i].tgt_hp;
        if (best < 0 || k1 < bk1 || (k1 == bk1 && k2 < bk2)) {
            best = static_cast<int>(i);
            bk1 = k1;
            bk2 = k2;
        }
    }
    return best;
}

// _pick: first move whose meta flag matches, with optional accuracy floor and
// effectiveness requirement (skip mult == 0).
int pick(const std::vector<Facts>& facts, bool MoveMeta::* flag, double min_acc,
         bool need_effective) {
    for (size_t i = 0; i < facts.size(); ++i) {
        const Facts& f = facts[i];
        if (!f.is_move || !(f.mm->*flag)) continue;
        if (f.mm->accuracy < min_acc) continue;
        if (need_effective && f.mult == 0.0) continue;
        return static_cast<int>(i);
    }
    return -1;
}

int first_max_damage(const std::vector<Facts>& facts) {  // first-maximal move by smart_dmg
    int best = -1;
    double bv = 0.0;
    for (size_t i = 0; i < facts.size(); ++i) {
        if (!facts[i].is_move) continue;
        double v = smart_dmg(facts[i]);
        if (best < 0 || v > bv) {
            best = static_cast<int>(i);
            bv = v;
        }
    }
    return best;
}

}  // namespace

int select_heuristic(const Battle& b, int player, Observer* obs, int kind) {
    const Side& me = (player == 0) ? b.p1 : b.p2;
    const Side& op = (player == 0) ? b.p2 : b.p1;
    if (obs) obs->mons |= static_cast<uint8_t>(1u << op.active);  // same reveal as build_state

    std::vector<Choice> choices;
    std::vector<Facts> facts;
    build_facts(b, player, choices, facts);
    if (facts.empty()) throw std::runtime_error("select_heuristic: no available actions");

    bool any_move = false;
    for (const Facts& f : facts) any_move |= f.is_move;

    if (kind == 0) {  // MaxDamagePolicy: base_power * mult * stab, no accuracy/fixed
        int best = -1;
        double bv = 0.0;
        for (size_t i = 0; i < facts.size(); ++i) {
            if (!facts[i].is_move) continue;
            double v = facts[i].mm->base_power * facts[i].mult * (facts[i].stab ? 1.5 : 1.0);
            if (best < 0 || v > bv) {
                best = static_cast<int>(i);
                bv = v;
            }
        }
        if (best >= 0 && bv > 0.0) return best;
        for (size_t i = 0; i < facts.size(); ++i)
            if (!facts[i].is_move) return static_cast<int>(i);
        return 0;
    }

    const Pokemon& my = me.mon();
    const Pokemon& opp = op.mon();
    const SpeciesMeta& my_sp = meta(my.species);
    const SpeciesMeta& op_sp = meta(opp.species);

    if (!any_move) {  // forced switch: safest, healthiest body
        int sw = best_switch(facts);
        return sw >= 0 ? sw : 0;
    }

    int best = first_max_damage(facts);
    double best_val = smart_dmg(facts[best]);
    bool opp_statused = opp.status != Status::None;

    if (kind == 1) {  // SmartHeuristicPolicy
        bool likely_ko = opp.hp_fraction() < 0.35 && best_val >= 80.0;
        if (facts[best].mm->h_hyperbeam && !likely_ko) {
            // Prefer the best non-recharge move when Hyper Beam won't KO — unless it gives
            // up too much damage (the 0.6 rule).
            int alt = -1;
            double av = 0.0;
            for (size_t i = 0; i < facts.size(); ++i) {
                if (!facts[i].is_move || facts[i].mm->h_hyperbeam) continue;
                double v = smart_dmg(facts[i]);
                if (alt < 0 || v > av) {
                    alt = static_cast<int>(i);
                    av = v;
                }
            }
            if (alt >= 0 && av >= 0.6 * best_val) {
                best = alt;
                best_val = av;
                likely_ko = opp.hp_fraction() < 0.35 && best_val >= 80.0;
            }
        }
        if (likely_ko) return best;
        if (!opp_statused && opp.hp_fraction() > 0.55) {
            int slp = pick(facts, &MoveMeta::h_sleep, 0.6, false);
            if (slp >= 0) return slp;
        }
        int op_speed = std::max(op_sp.speed, 0), my_speed = std::max(my_sp.speed, 0);
        if (!opp_statused && op_speed > my_speed) {  // raw base speeds, like the Python
            int par = pick(facts, &MoveMeta::h_para, 0.0, true);
            if (par >= 0) return par;
        }
        if (my.hp_fraction() < 0.45) {
            int heal = pick(facts, &MoveMeta::h_heal, 0.0, false);
            if (heal >= 0) return heal;
        }
        double base = facts[best].mm->base_power;
        if (base > 0.0 && best_val <= 0.5 * base && best_val < 60.0) {
            int sw = best_switch(facts);
            if (sw >= 0 && incoming_or_1(facts[sw]) < 1.0) return sw;
        }
        return best;
    }

    // kind == 2: StallerPolicy
    if (opp.hp_fraction() < 0.30 && best_val >= 70.0) return best;
    bool foe_asleep = false;  // over REVEALED opp mons, like state.opponent_team
    for (int i = 0; i < static_cast<int>(op.team.size()); ++i)
        if (revealed(obs, i) && op.team[i].status == Status::Sleep) foe_asleep = true;
    if (!opp_statused && opp.hp_fraction() > 0.5 && !foe_asleep) {
        int slp = pick(facts, &MoveMeta::h_sleep, 0.55, false);
        if (slp >= 0) return slp;
    }
    if (!opp_statused) {
        int par = pick(facts, &MoveMeta::h_para, 0.0, true);
        if (par >= 0) return par;
    }
    if (my.hp_fraction() < 0.65) {  // RECOVER_BELOW
        int heal = pick(facts, &MoveMeta::h_heal, 0.0, false);
        if (heal >= 0) return heal;
    }
    double base = facts[best].mm->base_power;
    if (base > 0.0 && best_val <= 0.5 * base) {
        int sw = best_switch(facts);
        if (sw >= 0 && incoming_or_1(facts[sw]) < 1.0) return sw;
    }
    return best;
}

void search_leaves(const Battle& b, int player, const std::vector<int>& my_cands,
                   const std::vector<int>& opp_cands, int rollouts,
                   const std::vector<uint64_t>& seeds, const Observer* obs,
                   int mode, int rkind, int depth, int cap,
                   float* rows, double* direct) {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    size_t l = 0;
    for (int i : my_cands) {
        for (int j : opp_cands) {
            for (int r = 0; r < rollouts; ++r, ++l) {
                Battle c = b;
                c.reseed(seeds.at(l));  // fresh dice — never replay the live game's RNG
                auto cm = c.choices(player), co = c.choices(1 - player);
                const Choice& mine = cm.at(static_cast<size_t>(i));
                const Choice& theirs = co.at(static_cast<size_t>(j));
                if (player == 0) c.step(mine, theirs);
                else c.step(theirs, mine);

                if (mode == 1) {  // playout to terminal (== _playout_value_fast)
                    Observer o1, o2;
                    int turns = 0;
                    while (c.result() == Result::Ongoing && turns < cap) {
                        ++turns;
                        int i1 = select_heuristic(c, 0, &o1, rkind);
                        int i2 = select_heuristic(c, 1, &o2, rkind);
                        step_pair(c, i1, i2, &o1, &o2);
                    }
                    Result res = c.result();
                    direct[l] = (res == Result::Tie || res == Result::Ongoing) ? 0.5
                              : ((res == Result::P1Win) == (player == 0) ? 1.0 : 0.0);
                    continue;
                }
                if (mode == 2) {  // bootstrap: depth heuristic turns (== _playout_bootstrap_fast)
                    for (int d = 0; d < depth && c.result() == Result::Ongoing; ++d) {
                        int i1 = select_heuristic(c, 0, nullptr, rkind);
                        int i2 = select_heuristic(c, 1, nullptr, rkind);
                        c.step(c.choices(0).at(i1), c.choices(1).at(i2));
                    }
                    Result res = c.result();
                    if (res != Result::Ongoing) {
                        direct[l] = res == Result::Tie ? 0.5
                                  : ((res == Result::P1Win) == (player == 0) ? 1.0 : 0.0);
                        continue;
                    }
                    direct[l] = kNaN;  // bootstrap leaves are full-information (obs ignored)
                    encode(c, player, nullptr, rows + l * kGlobalDim, nullptr);
                    continue;
                }
                // mode 0: value leaf — encode the post-step state with a per-leaf obs snapshot
                // (encode mutates it: the opponent's possibly-new active becomes revealed).
                direct[l] = kNaN;
                Observer tmp;
                Observer* leaf_obs = nullptr;
                if (obs) {
                    tmp = *obs;
                    leaf_obs = &tmp;
                }
                encode(c, player, leaf_obs, rows + l * kGlobalDim, nullptr);
            }
        }
    }
    if (l != seeds.size()) throw std::invalid_argument("search_leaves: seeds length mismatch");
}

}  // namespace cinnabar::enc
