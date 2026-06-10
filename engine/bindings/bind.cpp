// pybind11 bindings — drive the Cinnabar Gen 1 engine from Python.
// Minimal surface for now: build a battle, query legal choices, step, read result.
// Enough for the smoke test and the upcoming differential harness / RL adapter.
#include <pybind11/numpy.h>  // the C++ encoder returns feature arrays
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // vector / pair / string conversions

#include <string>
#include <vector>

#include "cinnabar/encoder.hpp"
#include "cinnabar/engine.hpp"

namespace py = pybind11;
using namespace cinnabar;

// Uppercase status code for the RL adapter ("" = no status). Gen 1 has no Toxic.
static std::string status_up(Status s) {
    switch (s) {
        case Status::Sleep: return "SLP";
        case Status::Poison: return "PSN";
        case Status::Burn: return "BRN";
        case Status::Freeze: return "FRZ";
        case Status::Paralysis: return "PAR";
        default: return "";
    }
}

PYBIND11_MODULE(cinnabar_engine, m) {
    m.doc() = "Cinnabar Gen 1 battle engine (C++)";

    py::enum_<Result>(m, "Result")
        .value("Ongoing", Result::Ongoing)
        .value("P1Win", Result::P1Win)
        .value("P2Win", Result::P2Win)
        .value("Tie", Result::Tie);

    py::enum_<ChoiceKind>(m, "ChoiceKind")
        .value("Move", ChoiceKind::Move)
        .value("Switch", ChoiceKind::Switch)
        .value("Pass", ChoiceKind::Pass);

    py::class_<Choice>(m, "Choice")
        .def_readonly("kind", &Choice::kind)
        .def_readonly("index", &Choice::index)
        .def("__repr__", [](const Choice& c) {
            const char* k = c.kind == ChoiceKind::Move ? "Move"
                          : c.kind == ChoiceKind::Switch ? "Switch" : "Pass";
            return "<Choice " + std::string(k) + " " + std::to_string(c.index) + ">";
        });

    py::class_<Battle>(m, "Battle")
        .def_readonly("turn", &Battle::turn)
        .def("choices", &Battle::choices, py::arg("player"))
        .def("step", &Battle::step, py::arg("c1"), py::arg("c2"))
        .def("result", &Battle::result)
        .def("set_clauses", &Battle::set_clauses, py::arg("on"))  // OU Sleep+Freeze Clause (training)
        .def("clone", [](const Battle& b) { return b; })          // deep copy for decision-time search
        .def("reseed", &Battle::reseed, py::arg("seed"))          // fresh dice for search rollouts
        // State injection (browser ground-truth reconstruction): set one mon's mid-battle
        // condition. status is the adapter's code ("", "SLP", "PAR", "FRZ", "BRN", "PSN", "TOX");
        // stages are -6..+6; modified stats are recomputed (stages, then burn/para drop).
        .def("set_mon_state", [](Battle& b, int player, int slot, double hp_fraction,
                                 const std::string& status, int sleep_turns, bool status_by_foe,
                                 int boost_atk, int boost_def, int boost_spc, int boost_spe,
                                 bool reflect, bool light_screen, bool must_recharge,
                                 int tox_stage, int confuse_turns, bool leech_seeded) {
            Status st = Status::None;
            bool toxic = false;
            if (status == "SLP") st = Status::Sleep;
            else if (status == "PAR") st = Status::Paralysis;
            else if (status == "BRN") st = Status::Burn;
            else if (status == "FRZ") st = Status::Freeze;
            else if (status == "PSN") st = Status::Poison;
            else if (status == "TOX") { st = Status::Poison; toxic = true; }
            set_mon_state(b, player, slot, hp_fraction, st, sleep_turns, status_by_foe,
                          boost_atk, boost_def, boost_spc, boost_spe, reflect, light_screen,
                          must_recharge, toxic, tox_stage, confuse_turns, leech_seeded);
        }, py::arg("player"), py::arg("slot"), py::arg("hp_fraction"), py::arg("status") = "",
           py::arg("sleep_turns") = 0, py::arg("status_by_foe") = true,
           py::arg("boost_atk") = 0, py::arg("boost_def") = 0, py::arg("boost_spc") = 0,
           py::arg("boost_spe") = 0, py::arg("reflect") = false, py::arg("light_screen") = false,
           py::arg("must_recharge") = false, py::arg("tox_stage") = 0,
           py::arg("confuse_turns") = 0, py::arg("leech_seeded") = false)
        // Choose the active slot (sets must_switch when that mon is fainted and a bench remains).
        .def("set_active_slot", [](Battle& b, int player, int slot) {
            set_active_slot(b, player, slot);
        }, py::arg("player"), py::arg("slot"))
        .def("active_species", [](const Battle& b, int player) {
            return (player == 0 ? b.p1 : b.p2).mon().species->name;
        }, py::arg("player"))
        .def("active_hp_fraction", [](const Battle& b, int player) {
            return (player == 0 ? b.p1 : b.p2).mon().hp_fraction();
        }, py::arg("player"))
        .def("active_hp", [](const Battle& b, int player) {
            return (player == 0 ? b.p1 : b.p2).mon().hp;
        }, py::arg("player"))
        .def("active_max_hp", [](const Battle& b, int player) {
            return (player == 0 ? b.p1 : b.p2).mon().max_hp;
        }, py::arg("player"))
        .def("active_status", [](const Battle& b, int player) {
            const auto& m = (player == 0 ? b.p1 : b.p2).mon();
            switch (m.status) {
                case Status::None: return std::string("none");
                case Status::Sleep: return std::string("slp");
                case Status::Poison: return std::string(m.toxic ? "tox" : "psn");
                case Status::Burn: return std::string("brn");
                case Status::Freeze: return std::string("frz");
                case Status::Paralysis: return std::string("par");
            }
            return std::string("?");
        }, py::arg("player"))
        // Per-mon team view (team order): (species, hp_fraction, status, fainted, active).
        // Drives the RL adapter's BattleState (team aggregates, switch targets, active views).
        .def("team_state", [](const Battle& b, int player) {
            const Side& s = (player == 0) ? b.p1 : b.p2;
            py::list out;
            for (int i = 0; i < static_cast<int>(s.team.size()); ++i) {
                const Pokemon& mon = s.team[i];
                out.append(py::make_tuple(mon.species->name, mon.hp_fraction(),
                                          status_up(mon.status), mon.fainted(), i == s.active));
            }
            return out;
        }, py::arg("player"))
        .def("must_switch", [](const Battle& b, int player) {
            return (player == 0 ? b.p1 : b.p2).must_switch;
        }, py::arg("player"))
        // Active-mon volatile state for the RL observation: stat stages, the Hyper Beam
        // recharge flag (a free turn the opponent owes), and remaining forced-sleep turns.
        .def("active_boosts", [](const Battle& b, int player) {
            const Pokemon& m = (player == 0 ? b.p1 : b.p2).mon();
            return py::make_tuple(m.boost_atk, m.boost_def, m.boost_spc, m.boost_spe);
        }, py::arg("player"))
        .def("active_must_recharge", [](const Battle& b, int player) {
            return (player == 0 ? b.p1 : b.p2).mon().must_recharge;
        }, py::arg("player"))
        .def("active_sleep_turns", [](const Battle& b, int player) {
            const Pokemon& m = (player == 0 ? b.p1 : b.p2).mon();
            return m.status == Status::Sleep ? m.sleep_turns : 0;
        }, py::arg("player"))
        // More volatiles for the RL observation: (confused, reflect, light_screen, leech_seeded,
        // disabled, toxic, tox_stage). Lets the agent perceive mechanics it otherwise can't see.
        .def("active_volatiles", [](const Battle& b, int player) {
            const Pokemon& m = (player == 0 ? b.p1 : b.p2).mon();
            return py::make_tuple(m.confuse_turns > 0, m.reflect, m.light_screen, m.leech_seeded,
                                  m.disable_turns > 0, m.toxic, m.tox_stage);
        }, py::arg("player"));

    // team1/team2 are list[tuple[str species, list[str] moves]].
    m.def("make_battle", &make_battle, py::arg("team1"), py::arg("team2"), py::arg("seed"));

    // ---- C++ observation encoder (see include/cinnabar/encoder.hpp) ------------------------
    // Static data is registered from Python (the same poke-env tables build_state uses), so
    // the encoder is pure arithmetic on identical inputs; a pytest asserts exact parity.
    m.attr("GLOBAL_DIM") = enc::kGlobalDim;
    m.attr("ACTION_DIM") = enc::kActionDim;

    py::class_<enc::Observer>(m, "Observer")
        .def(py::init<>())
        .def("clone", [](const enc::Observer& o) { return o; })  // snapshot for search leaves
        .def_property_readonly("mons", [](const enc::Observer& o) { return int(o.mons); })
        .def("move_mask", [](const enc::Observer& o, int slot) {
            return int(o.moves.at(static_cast<size_t>(slot)));
        }, py::arg("slot"))
        // Setters so search can convert its Python Reveal memory into an Observer.
        .def("reveal_mon", [](enc::Observer& o, int slot) {
            o.mons |= static_cast<uint8_t>(1u << slot);
        }, py::arg("slot"))
        .def("see_move", [](enc::Observer& o, int slot, int move_slot) {
            o.moves.at(static_cast<size_t>(slot)) |= static_cast<uint8_t>(1u << move_slot);
        }, py::arg("slot"), py::arg("move_slot"));

    // chart: 15x15 [defender][attacker] floats; moves: id -> 16-tuple matching MoveMeta field
    // order; species: id -> (type indices, base speed). See engine_cpp.register_encoder.
    m.def("register_encoder", [](const std::vector<std::vector<double>>& chart,
                                 const py::dict& moves, const py::dict& species) {
        if (chart.size() != enc::kTypes) throw std::invalid_argument("chart must be 15x15");
        double c[enc::kTypes][enc::kTypes];
        for (int d = 0; d < enc::kTypes; ++d) {
            if (chart[d].size() != enc::kTypes) throw std::invalid_argument("chart must be 15x15");
            for (int a = 0; a < enc::kTypes; ++a) c[d][a] = chart[d][a];
        }
        enc::register_type_chart(c);
        for (auto item : moves) {
            auto t = item.second.cast<py::tuple>();
            enc::MoveMeta mm;
            mm.base_power = t[0].cast<double>();
            mm.type_idx = t[1].cast<int>();
            mm.is_status = t[2].cast<bool>();
            mm.accuracy = t[3].cast<double>();
            mm.fixed = t[4].cast<double>();
            mm.effect_status = t[5].cast<int>();
            mm.effect_chance = t[6].cast<double>();
            mm.heals = t[7].cast<bool>();
            mm.boosts_self = t[8].cast<bool>();
            mm.lowers_foe = t[9].cast<bool>();
            mm.recharge = t[10].cast<bool>();
            mm.self_destruct = t[11].cast<bool>();
            mm.h_sleep = t[12].cast<bool>();
            mm.h_para = t[13].cast<bool>();
            mm.h_heal = t[14].cast<bool>();
            mm.h_hyperbeam = t[15].cast<bool>();
            enc::register_move(item.first.cast<std::string>(), mm);
        }
        for (auto item : species) {
            auto t = item.second.cast<py::tuple>();
            auto types = t[0].cast<std::vector<int>>();
            enc::SpeciesMeta sm;
            for (size_t i = 0; i < types.size() && i < 2; ++i)
                sm.types[i] = static_cast<int8_t>(types[i]);
            sm.speed = t[1].cast<int>();
            enc::register_species(item.first.cast<std::string>(), sm);
        }
    }, py::arg("chart"), py::arg("moves"), py::arg("species"));
    m.def("encoder_ready", &enc::encoder_ready);

    // Encode one player's view -> (global float32[G], actions float32[N, A]). Mutates obs
    // (reveals the opponent's active) exactly like build_state mutates Reveal.
    m.def("encode", [](const Battle& b, int player, enc::Observer* obs) {
        size_t n = b.choices(player).size();
        py::array_t<float> g(py::array::ShapeContainer{(py::ssize_t)enc::kGlobalDim});
        py::array_t<float> a(py::array::ShapeContainer{(py::ssize_t)n, (py::ssize_t)enc::kActionDim});
        enc::encode(b, player, obs, g.mutable_data(), n ? a.mutable_data() : nullptr);
        return py::make_tuple(g, a);
    }, py::arg("battle"), py::arg("player"), py::arg("observer") = nullptr);

    // The training fast path: encode B battles in one call ->
    // (glob [B,G] f32, act [B,K,A] f32, mask [B,K] bool, my_mat [B] f64, opp_mat [B] f64),
    // K = max action count, padded rows zeroed/masked like train_engine._pad.
    m.def("encode_batch", [](const py::sequence& battles, const py::sequence& players,
                             const py::sequence& observers) {
        size_t n = py::len(battles);
        if (py::len(players) != n || py::len(observers) != n)
            throw std::invalid_argument("encode_batch: length mismatch");
        std::vector<Battle*> bs(n);
        std::vector<int> ps(n);
        std::vector<enc::Observer*> os(n);
        size_t kmax = 1;
        for (size_t i = 0; i < n; ++i) {
            bs[i] = battles[i].cast<Battle*>();
            ps[i] = players[i].cast<int>();
            py::object o = observers[i];
            os[i] = o.is_none() ? nullptr : o.cast<enc::Observer*>();
            kmax = std::max(kmax, bs[i]->choices(ps[i]).size());
        }
        py::array_t<float> glob(py::array::ShapeContainer{(py::ssize_t)n, (py::ssize_t)enc::kGlobalDim});
        py::array_t<float> act(py::array::ShapeContainer{(py::ssize_t)n, (py::ssize_t)kmax,
                                                         (py::ssize_t)enc::kActionDim});
        py::array_t<bool> mask(py::array::ShapeContainer{(py::ssize_t)n, (py::ssize_t)kmax});
        py::array_t<double> my_mat(py::array::ShapeContainer{(py::ssize_t)n});
        py::array_t<double> opp_mat(py::array::ShapeContainer{(py::ssize_t)n});
        std::fill(act.mutable_data(), act.mutable_data() + n * kmax * enc::kActionDim, 0.0f);
        std::fill(mask.mutable_data(), mask.mutable_data() + n * kmax, false);
        for (size_t i = 0; i < n; ++i) {
            int na = enc::encode(*bs[i], ps[i], os[i],
                                 glob.mutable_data() + i * enc::kGlobalDim,
                                 act.mutable_data() + i * kmax * enc::kActionDim,
                                 my_mat.mutable_data() + i, opp_mat.mutable_data() + i);
            bool* mrow = mask.mutable_data() + i * kmax;
            for (int k = 0; k < na; ++k) mrow[k] = true;
        }
        return py::make_tuple(glob, act, mask, my_mat, opp_mat);
    }, py::arg("battles"), py::arg("players"), py::arg("observers"));

    // Record both sides' selected moves into the other side's observer, then step.
    m.def("step_pair", &enc::step_pair, py::arg("battle"), py::arg("i1"), py::arg("i2"),
          py::arg("obs1") = nullptr, py::arg("obs2") = nullptr);

    // C++ heuristic pilots (parity-tested vs policy.py). Returns the chosen choice index.
    auto kind_id = [](const std::string& kind) {
        int k = kind == "maxdamage" ? 0 : kind == "smart" ? 1 : kind == "staller" ? 2 : -1;
        if (k < 0) throw std::invalid_argument("kind must be maxdamage|smart|staller");
        return k;
    };
    m.def("select_heuristic", [kind_id](const Battle& b, int player, enc::Observer* obs,
                                        const std::string& kind) {
        return enc::select_heuristic(b, player, obs, kind_id(kind));
    }, py::arg("battle"), py::arg("player"), py::arg("observer") = nullptr, py::arg("kind") = "maxdamage");

    // Decision-time search's leaf loop in one call (see encoder.hpp). Returns
    // (rows [L, G] f32, direct [L] f64) — direct[l] is NaN where rows[l] should be scored
    // by the value head; L = len(my_cands) * len(opp_cands) * rollouts.
    m.def("search_leaves", [kind_id](const Battle& b, int player, const std::vector<int>& my_cands,
                                     const std::vector<int>& opp_cands, int rollouts,
                                     const std::vector<uint64_t>& seeds, const enc::Observer* obs,
                                     int mode, const std::string& rkind, int depth, int cap) {
        size_t L = my_cands.size() * opp_cands.size() * static_cast<size_t>(rollouts);
        if (seeds.size() != L) throw std::invalid_argument("search_leaves: need one seed per leaf");
        py::array_t<float> rows(py::array::ShapeContainer{(py::ssize_t)L, (py::ssize_t)enc::kGlobalDim});
        py::array_t<double> direct(py::array::ShapeContainer{(py::ssize_t)L});
        enc::search_leaves(b, player, my_cands, opp_cands, rollouts, seeds, obs,
                           mode, kind_id(rkind), depth, cap,
                           rows.mutable_data(), direct.mutable_data());
        return py::make_tuple(rows, direct);
    }, py::arg("battle"), py::arg("player"), py::arg("my_cands"), py::arg("opp_cands"),
       py::arg("rollouts"), py::arg("seeds"), py::arg("observer") = nullptr,
       py::arg("mode") = 0, py::arg("rkind") = "smart", py::arg("depth") = 0, py::arg("cap") = 150);
}
