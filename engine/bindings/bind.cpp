// pybind11 bindings — drive the Cinnabar Gen 1 engine from Python.
// Minimal surface for now: build a battle, query legal choices, step, read result.
// Enough for the smoke test and the upcoming differential harness / RL adapter.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // vector / pair / string conversions

#include <string>

#include "cinnabar/engine.hpp"

namespace py = pybind11;
using namespace cinnabar;

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
        }, py::arg("player"));

    // team1/team2 are list[tuple[str species, list[str] moves]].
    m.def("make_battle", &make_battle, py::arg("team1"), py::arg("team2"), py::arg("seed"));
}
