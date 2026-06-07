// Reference trace(s) from Showdown's own gen1 sim (the submodule), driven deterministically.
// 1v1, both sides use "move 1" every turn, fixed Gen-5 RNG seed. Dumps per-turn JSON of exact
// HP/status so our engine can be diffed bit-for-bit.
//
//   node ref_trace.js <s0> <s1> <s2> <s3>   one battle (set REF_DEBUG=1 for stats + protocol log)
//   node ref_trace.js --sweep [N]           N battles over deterministic seeds (default 100)
const path = require("path");

// The built Showdown package (submodule). Adjust if your layout differs.
const sim = require(path.resolve(__dirname, "../../server/pokemon-showdown"));
const { Teams, Battle } = sim;

// Max EVs so Showdown uses max-StatExp stats (what real Gen 1 OU uses), matching the
// engine's stat formula. Without this, customgame defaults to StatExp 0 (Tauros HP 290).
const EVS = "EVs: 252 HP / 252 Atk / 252 Def / 252 SpA / 252 SpD / 252 Spe";

// Gen 1 has NO abilities, but gen1customgame otherwise assigns each species its modern
// default ability and FIRES it (e.g. Tauros's Intimidate drops the foe's Attack),
// contaminating the diff. Strip the ability from every set so the reference is true Gen 1.
function packTeam(text) {
    const team = Teams.import(text);
    for (const set of team) set.ability = "No Ability";
    return Teams.pack(team);
}
const P1 = packTeam(`Tauros\n${EVS}\n- Earthquake\n`);
const P2 = packTeam(`Snorlax\n${EVS}\n- Earthquake\n`);

const code = (s) => s || "none"; // '' -> none; else slp/par/brn/frz/psn

function runBattle(seedWords, debug) {
    const battle = new Battle({
        formatid: "gen1customgame",
        seed: seedWords,
        p1: { name: "p1", team: P1 },
        p2: { name: "p2", team: P2 },
    });
    if (debug) {
        for (const [label, side] of [["p1", battle.sides[0]], ["p2", battle.sides[1]]]) {
            const p = side.active[0];
            console.error(`${label} ${p.set.species} L${p.level} maxhp=${p.maxhp} ability=${p.ability || "(none)"}`);
            console.error(`   storedStats = ${JSON.stringify(p.storedStats)}`);
        }
    }
    const snap = () => {
        const a = battle.sides[0].active[0];
        const b = battle.sides[1].active[0];
        return {
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
    if (debug) {
        console.error("\n--- protocol log ---");
        console.error(battle.log.join("\n"));
    }
    return { winner: battle.winner || null, trace };
}

// Deterministic 4x16-bit Gen-5 seed for sweep index i (an LCG over the index).
function seedFor(i) {
    let x = (i + 1) >>> 0;
    const w = [];
    for (let k = 0; k < 4; k++) {
        x = (Math.imul(x, 1103515245) + 12345) >>> 0;
        w.push((x >>> 16) & 0xFFFF);
    }
    return w;
}

const args = process.argv.slice(2);
if (args[0] === "--sweep") {
    const n = parseInt(args[1] || "100", 10);
    const out = [];
    for (let i = 0; i < n; i++) {
        const words = seedFor(i);
        out.push({ words, ...runBattle(words, false) });
    }
    console.log(JSON.stringify({ sweep: out }));
} else {
    const seed = args.slice(0, 4).map(Number);
    if (seed.length !== 4 || seed.some(Number.isNaN)) {
        console.error("usage: node ref_trace.js <s0> <s1> <s2> <s3>   |   node ref_trace.js --sweep [N]");
        process.exit(1);
    }
    console.log(JSON.stringify(runBattle(seed, !!process.env.REF_DEBUG)));
}
