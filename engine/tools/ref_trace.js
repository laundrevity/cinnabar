// Reference trace from Showdown's own gen1 sim (the submodule), driven deterministically.
// 1v1, both sides use "move 1" every turn, fixed Gen-5 RNG seed. Dumps a per-turn JSON
// trace of exact HP/status so our engine can be diffed against it bit-for-bit.
//
//   node ref_trace.js <s0> <s1> <s2> <s3>     (four 16-bit seed words)
//
// Outputs one JSON line: {"winner":..., "trace":[{turn,p1_hp,p1_maxhp,p1_status,...}, ...]}.
const path = require("path");

// The built Showdown package (submodule). Adjust if your layout differs.
const sim = require(path.resolve(__dirname, "../../server/pokemon-showdown"));
const { Teams, Battle } = sim;

const P1 = Teams.pack(Teams.import("Tauros\n- Earthquake\n"));
const P2 = Teams.pack(Teams.import("Snorlax\n- Earthquake\n"));

const seed = process.argv.slice(2, 6).map(Number);
if (seed.length !== 4 || seed.some(Number.isNaN)) {
    console.error("usage: node ref_trace.js <s0> <s1> <s2> <s3>");
    process.exit(1);
}

const battle = new Battle({
    formatid: "gen1customgame",
    seed,
    p1: { name: "p1", team: P1 },
    p2: { name: "p2", team: P2 },
});

const code = (s) => s || "none"; // '' -> none; else slp/par/brn/frz/psn
const snap = () => {
    const a = battle.sides[0].active[0];
    const b = battle.sides[1].active[0];
    return {
        turn: battle.turn,
        p1_hp: a.hp, p1_maxhp: a.maxhp, p1_status: code(a.status),
        p2_hp: b.hp, p2_maxhp: b.maxhp, p2_status: code(b.status),
    };
};

const trace = [];
let guard = 0;
while (!battle.ended && guard++ < 2000) {
    battle.makeChoices("move 1", "move 1");
    trace.push(snap());
}
console.log(JSON.stringify({ winner: battle.winner || null, trace }));
