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

// --- provisional scoped data (to be generated from Showdown) ---
static const Species ALAKAZAM{"Alakazam", Type::Psychic, Type::None, 55, 50, 45, 135, 120};
static const Species CHANSEY{"Chansey", Type::Normal, Type::None, 250, 5, 5, 105, 50};
static const Species EXEGGUTOR{"Exeggutor", Type::Grass, Type::Psychic, 95, 95, 85, 125, 55};
static const Species SNORLAX{"Snorlax", Type::Normal, Type::None, 160, 110, 65, 65, 30};
static const Species TAUROS{"Tauros", Type::Normal, Type::None, 75, 100, 95, 70, 110};
static const Species STARMIE{"Starmie", Type::Water, Type::Psychic, 60, 75, 85, 100, 115};

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

static Side make_team() {
    return Side{{
        make_pokemon(&ALAKAZAM, {&PSYCHIC, &THUNDER_WAVE, &RECOVER, &SEISMIC_TOSS}),
        make_pokemon(&CHANSEY, {&ICE_BEAM, &THUNDERBOLT, &THUNDER_WAVE, &SOFT_BOILED}),
        make_pokemon(&EXEGGUTOR, {&PSYCHIC, &SLEEP_POWDER, &EXPLOSION}),
        make_pokemon(&SNORLAX, {&BODY_SLAM, &EARTHQUAKE, &EXPLOSION, &REST}),
        make_pokemon(&TAUROS, {&BODY_SLAM, &EARTHQUAKE, &BLIZZARD, &HYPER_BEAM}),
        make_pokemon(&STARMIE, {&THUNDERBOLT, &ICE_BEAM, &RECOVER, &THUNDER_WAVE}),
    }};
}

int main() {
    // Gen 1 stats + damage formula + type chart (foundation).
    Pokemon t = make_pokemon(&TAUROS, {&BODY_SLAM});
    CHECK_EQ(t.max_hp, 353);
    CHECK_EQ(t.atk, 298);
    CHECK_EQ(t.spe, 318);
    CHECK_EQ(make_pokemon(&SNORLAX, {}).max_hp, 523);
    CHECK_EQ(gen1_damage(100, 100, 298, 298, false, 1.0, false, 255), 86);
    CHECK_EQ(gen1_damage(100, 100, 298, 298, false, 1.0, true, 255), 166);
    CHECK_EQ(gen1_damage(100, 100, 298, 298, true, 2.0, false, 255), 258);
    CHECK(type_effectiveness(Type::Ghost, Type::Psychic) == 0.0);   // Gen 1 bug
    CHECK(type_effectiveness(Type::Bug, Type::Poison) == 2.0);      // Gen 1 quirk
    CHECK(type_effectiveness(Type::Electric, Type::Ground) == 0.0);

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
        Battle b1(make_team(), make_team(), 6);
        Battle b2(make_team(), make_team(), 6);
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

    // Residual: poison ticks 1/16 max HP at end of turn.
    {
        Battle b(make_team(), make_team(), 7);
        b.p1.mon().status = Status::Poison;
        int max = b.p1.mon().max_hp, before = b.p1.mon().hp;
        b.step(pass_choice(), pass_choice());
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

    if (failures == 0) std::printf("ALL ENGINE TESTS PASSED\n");
    else std::printf("%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
