// Unit tests for the Cinnabar engine. Expected values are hand-computed; the
// authoritative fidelity check is differential testing against Showdown (next phase).
#include "cinnabar/engine.hpp"

#include <cstdio>
#include <vector>

using namespace cinnabar;

static int failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)
#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        long _a = (long)(a), _b = (long)(b);                                   \
        if (_a != _b) { std::printf("FAIL %s:%d  %s==%s (%ld vs %ld)\n", __FILE__, __LINE__, #a, #b, _a, _b); ++failures; } \
    } while (0)

// Species come from the generated gen1_data.hpp via species("Name"). Moves stay
// hand-coded here for now (move-effect codegen is a later slice).

static const MoveData PSYCHIC{"Psychic", Type::Psychic, Category::Special, 90, 100};
static const MoveData THUNDER_WAVE{"Thunder Wave", Type::Electric, Category::Status, 0, 100, 0, Effect::Paralyze, 100};
static const MoveData RECOVER{"Recover", Type::Normal, Category::Status, 0, 100, 0, Effect::Heal, 100};
static const MoveData SEISMIC_TOSS{"Seismic Toss", Type::Fighting, Category::Status, 0, 100, 100};
static const MoveData ICE_BEAM{"Ice Beam", Type::Ice, Category::Special, 95, 100, 0, Effect::Freeze, 10};
static const MoveData THUNDERBOLT{"Thunderbolt", Type::Electric, Category::Special, 95, 100, 0, Effect::Paralyze, 10};
static const MoveData SOFT_BOILED{"Soft-Boiled", Type::Normal, Category::Status, 0, 100, 0, Effect::Heal, 100};
static const MoveData SLEEP_POWDER{"Sleep Powder", Type::Grass, Category::Status, 0, 75, 0, Effect::Sleep, 100};
static const MoveData EXPLOSION{"Explosion", Type::Normal, Category::Physical, 170, 100, 0, Effect::SelfDestruct, 0};
static const MoveData BODY_SLAM{"Body Slam", Type::Normal, Category::Physical, 85, 100, 0, Effect::Paralyze, 30};
static const MoveData EARTHQUAKE{"Earthquake", Type::Ground, Category::Physical, 100, 100};
static const MoveData BLIZZARD{"Blizzard", Type::Ice, Category::Special, 120, 90, 0, Effect::Freeze, 10};
static const MoveData HYPER_BEAM{"Hyper Beam", Type::Normal, Category::Physical, 150, 90};  // recharge not modelled
static const MoveData REST{"Rest", Type::Psychic, Category::Status, 0, 100, 0, Effect::Rest, 100};
static const MoveData AMNESIA{"Amnesia", Type::Psychic, Category::Status, 0, 100, 0, Effect::None, 0, 2, 2, false, 0};
static const MoveData SWORDS_DANCE{"Swords Dance", Type::Normal, Category::Status, 0, 100, 0, Effect::None, 0, 0, 2, false, 0};

static Side make_team() {
    return Side{{
        make_pokemon(&species("Alakazam"), {&PSYCHIC, &THUNDER_WAVE, &RECOVER, &SEISMIC_TOSS}),
        make_pokemon(&species("Chansey"), {&ICE_BEAM, &THUNDERBOLT, &THUNDER_WAVE, &SOFT_BOILED}),
        make_pokemon(&species("Exeggutor"), {&PSYCHIC, &SLEEP_POWDER, &EXPLOSION}),
        make_pokemon(&species("Snorlax"), {&BODY_SLAM, &EARTHQUAKE, &EXPLOSION, &REST}),
        make_pokemon(&species("Tauros"), {&BODY_SLAM, &EARTHQUAKE, &BLIZZARD, &HYPER_BEAM}),
        make_pokemon(&species("Starmie"), {&THUNDERBOLT, &ICE_BEAM, &RECOVER, &THUNDER_WAVE}),
    }};
}

int main() {
    // Gen 1 stats + damage formula + type chart (foundation).
    Pokemon t = make_pokemon(&species("Tauros"), {&BODY_SLAM});
    CHECK_EQ(t.max_hp, 353);
    CHECK_EQ(t.atk, 298);
    CHECK_EQ(t.spe, 318);
    CHECK_EQ(make_pokemon(&species("Snorlax"), {}).max_hp, 523);
    CHECK_EQ(gen1_damage(100, 100, 298, 298, false, Type::Normal, Type::Normal, Type::None, false, 255), 86);
    CHECK_EQ(gen1_damage(100, 100, 298, 298, false, Type::Normal, Type::Normal, Type::None, true, 255), 166);
    CHECK_EQ(gen1_damage(100, 100, 298, 298, true, Type::Water, Type::Ground, Type::None, false, 255), 258);
    // Dual-type effectiveness is order-sensitive: Ice vs Water/Flying (resist then super)
    // floors to 160, but Flying/Water (super then resist) gives 161 — a combined ×1.0 misses this.
    CHECK_EQ(gen1_damage(100, 95, 200, 100, false, Type::Ice, Type::Water, Type::Flying, false, 255), 160);
    CHECK_EQ(gen1_damage(100, 95, 200, 100, false, Type::Ice, Type::Flying, Type::Water, false, 255), 161);
    CHECK(type_effectiveness(Type::Ghost, Type::Psychic) == 0.0);   // Gen 1 bug
    CHECK(type_effectiveness(Type::Bug, Type::Poison) == 2.0);      // Gen 1 quirk
    CHECK(type_effectiveness(Type::Electric, Type::Ground) == 0.0);

    // RNG bit-identical to Showdown's Gen 5 LCG (reference values from the LCG recurrence).
    {
        RNG r(0x123456789ABCDEF0ULL);
        CHECK_EQ(r.next(), 3683702347u);
        CHECK_EQ(r.next(), 3207779802u);
        CHECK_EQ(r.next(), 4072565397u);
        RNG r2(0x123456789ABCDEF0ULL);
        CHECK_EQ(r2.random(100), 85);
        RNG r3(0x123456789ABCDEF0ULL);
        CHECK_EQ(r3.range(217, 255), 250);  // == Showdown PRNG.random(217, 256)
    }

    // Stat stages: Amnesia (+2 Special) and Swords Dance (+2 Attack) each double the stat.
    {
        Side a{{make_pokemon(&species("Alakazam"), {&AMNESIA})}};
        Side b{{make_pokemon(&species("Snorlax"), {&BODY_SLAM})}};
        Battle bt(std::move(a), std::move(b), 1);
        int base_spc = bt.p1.mon().spc;
        bt.step(move_choice(0), pass_choice());  // Amnesia (self, +2 Special)
        CHECK_EQ(bt.p1.mon().boost_spc, 2);
        CHECK_EQ(bt.p1.mon().m_spc, base_spc * 2);
        CHECK_EQ(bt.p1.mon().spc, base_spc);     // stored stat is unchanged
    }
    {
        Side a{{make_pokemon(&species("Tauros"), {&SWORDS_DANCE})}};
        Side b{{make_pokemon(&species("Snorlax"), {&BODY_SLAM})}};
        Battle bt(std::move(a), std::move(b), 1);
        int base_atk = bt.p1.mon().atk;
        bt.step(move_choice(0), pass_choice());  // Swords Dance (self, +2 Attack)
        CHECK_EQ(bt.p1.mon().m_atk, base_atk * 2);
    }

    // Switching: active changes and the outgoing mon's Reflect clears.
    {
        Battle b(make_team(), make_team(), 1);
        b.p1.mon().reflect = true;
        Pokemon* leaving = &b.p1.mon();
        b.step(switch_choice(3), pass_choice());
        CHECK_EQ(b.p1.active, 3);
        CHECK(leaving->reflect == false);
    }

    // Thunder Wave paralyzes (deterministic: 100% effect, Electric hits Normal).
    {
        Battle b(make_team(), make_team(), 2);  // both lead Alakazam
        b.step(move_choice(1), pass_choice());  // p1 Alakazam Thunder Wave
        CHECK(b.p2.mon().status == Status::Paralysis);
    }

    // Rest: self-sleep (2 turns) + full heal.
    {
        Battle b(make_team(), make_team(), 3);
        b.p1.active = 3;             // Snorlax
        b.p1.mon().hp = 1;
        b.step(move_choice(3), pass_choice());  // Snorlax Rest
        CHECK_EQ(b.p1.mon().hp, b.p1.mon().max_hp);
        CHECK(b.p1.mon().status == Status::Sleep);
        CHECK_EQ(b.p1.mon().sleep_turns, 2);
    }

    // Recover heals ~50% (capped at max).
    {
        Battle b(make_team(), make_team(), 4);  // Alakazam has Recover at slot 2
        b.p1.mon().hp = 10;
        int max = b.p1.mon().max_hp;
        b.step(move_choice(2), pass_choice());
        CHECK_EQ(b.p1.mon().hp, std::min(max, 10 + max / 2));
    }

    // Explosion: user faints, target takes heavy damage.
    {
        Battle b(make_team(), make_team(), 5);
        b.p1.active = 3;  // Snorlax (Explosion at slot 2)
        int before = b.p2.mon().hp;
        b.step(move_choice(2), pass_choice());
        CHECK(b.p1.mon().fainted());
        CHECK(b.p2.mon().hp < before);
    }

    // Reflect halves physical damage taken (same seed, identical RNG draws).
    {
        Battle b1(make_team(), make_team(), 1);  // seed chosen so Body Slam doesn't crit
        Battle b2(make_team(), make_team(), 1);  // (a crit would ignore Reflect, see engine.cpp)
        b1.p1.active = b2.p1.active = 4;  // Tauros (fast) uses Body Slam (slot 0)
        b1.p2.active = b2.p2.active = 3;  // vs Snorlax (slow)
        b2.p2.mon().reflect = true;
        int max = b1.p2.mon().max_hp;
        b1.step(move_choice(0), pass_choice());
        b2.step(move_choice(0), pass_choice());
        int loss_plain = max - b1.p2.mon().hp;
        int loss_reflect = max - b2.p2.mon().hp;
        CHECK(loss_plain > 0);
        CHECK(loss_reflect > 0);
        CHECK(loss_reflect < loss_plain);
    }

    // Residual: poison ticks 1/16 max HP after the afflicted mon's own move (onAfterMoveSelf).
    {
        Battle b(make_team(), make_team(), 7);
        b.p1.mon().status = Status::Poison;
        int max = b.p1.mon().max_hp, before = b.p1.mon().hp;
        b.step(move_choice(0), pass_choice());  // p1 uses a move (no self-damage); poison ticks after it
        CHECK_EQ(before - b.p1.mon().hp, std::max(1, max / 16));
    }

    // Full 6v6 battles terminate with a winner under random legal play.
    {
        RNG pick(0xC0FFEE);
        auto choose = [&](const std::vector<Choice>& cs) {
            return cs.empty() ? pass_choice() : cs[pick.range(0, (int)cs.size() - 1)];
        };
        for (uint64_t seed = 1; seed <= 30; ++seed) {
            Battle b(make_team(), make_team(), seed);
            Result r = Result::Ongoing;
            for (int turn = 0; turn < 5000 && r == Result::Ongoing; ++turn)
                r = b.step(choose(b.choices(0)), choose(b.choices(1)));
            CHECK(r != Result::Ongoing);
        }
    }

    // move() table + make_battle from team specs (the API the Python bindings use).
    {
        CHECK_EQ(move("Body Slam").power, 85);
        CHECK_EQ(move("Seismic Toss").fixed, 100);
        TeamSpec spec = {
            {"Tauros", {"Body Slam", "Earthquake", "Blizzard", "Hyper Beam"}},
            {"Snorlax", {"Body Slam", "Earthquake", "Explosion", "Rest"}},
            {"Chansey", {"Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled"}},
        };
        RNG pick(99);
        auto choose = [&](const std::vector<Choice>& cs) {
            return cs.empty() ? pass_choice() : cs[pick.range(0, (int)cs.size() - 1)];
        };
        Battle b = make_battle(spec, spec, 123);
        Result r = Result::Ongoing;
        for (int t = 0; t < 5000 && r == Result::Ongoing; ++t)
            r = b.step(choose(b.choices(0)), choose(b.choices(1)));
        CHECK(r != Result::Ongoing);
    }

    // Sleep Clause (training flag, default off): a foe-inflicted sleep fails when another of that
    // side's mons is already asleep by a foe's move. Freeze Clause shares the same code path. The
    // clause is off by default, so the bit-for-bit fidelity harness (clause-free) is unaffected.
    {
        static const MoveData LULLABY{"Lullaby", Type::Normal, Category::Status, 0, 100, 0, Effect::Sleep, 100};
        auto setup = [&](bool clauses) {
            Side p1{{make_pokemon(&species("Alakazam"), {&LULLABY, &RECOVER})}};
            Side p2{{make_pokemon(&species("Chansey"), {&SOFT_BOILED}),
                     make_pokemon(&species("Snorlax"), {&BODY_SLAM})}};
            Battle b(std::move(p1), std::move(p2), 5);
            b.p2.team[1].status = Status::Sleep;     // a benched foe is already asleep...
            b.p2.team[1].status_by_foe = true;       // ...from a foe's move (counts for the clause)
            if (clauses) b.set_clauses(true);
            return b;
        };
        Battle on = setup(true);
        on.step(move_choice(0), move_choice(0));     // Alakazam Lullaby vs Chansey
        CHECK(on.p2.team[0].status != Status::Sleep);  // clause ON: the second sleep fails
        Battle off = setup(false);
        off.step(move_choice(0), move_choice(0));
        CHECK(off.p2.team[0].status == Status::Sleep); // clause OFF (default): the sleep lands
    }

    if (failures == 0) std::printf("ALL ENGINE TESTS PASSED\n");
    else std::printf("%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
