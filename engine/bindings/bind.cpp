// pybind11 bindings — drive the Cinnabar Gen 1 engine from Python.
// Minimal surface for now: build a battle, query legal choices, step, read result.
// Enough for the smoke test and the upcoming differential harness / RL adapter.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // vector / pair / string conversions

#include <string>

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
            switch ((player == 0 ? b.p1 : b.p2).mon().status) {
                case Status::None: return std::string("none");
                case Status::Sleep: return std::string("slp");
                case Status::Poison: return std::string("psn");
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
        }, py::arg("player"));

    // team1/team2 are list[tuple[str species, list[str] moves]].
    m.def("make_battle", &make_battle, py::arg("team1"), py::arg("team2"), py::arg("seed"));
}
