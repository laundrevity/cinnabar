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
// Matchup configurable via env. CINNABAR_P{1,2}_TEAM is a comma-separated species list (each
// gets CINNABAR_P{1,2}_MOVE); falls back to the single-species 1v1 vars.
function buildTeam(speciesList, moveName) {
    const text = speciesList.map((s) => `${s.trim()}\n${EVS}\n- ${moveName}\n`).join("\n");
    return packTeam(text);
}
const P1_TEAM = (process.env.CINNABAR_P1_TEAM || process.env.CINNABAR_P1_SPECIES || "Tauros").split(",");
const P2_TEAM = (process.env.CINNABAR_P2_TEAM || process.env.CINNABAR_P2_SPECIES || "Snorlax").split(",");
const P1M = process.env.CINNABAR_P1_MOVE || "Earthquake";
const P2M = process.env.CINNABAR_P2_MOVE || "Earthquake";
const P1 = buildTeam(P1_TEAM, P1M);
const P2 = buildTeam(P2_TEAM, P2M);

const code = (s) => s || "none"; // '' -> none; else slp/par/brn/frz/psn

function runBattle(seedWords, debug) {
    let battle;
    if (debug) {
        // Construct without players first so the PRNG can be wrapped before the battle starts,
        // capturing EVERY random() call (incl. battle-start / per-event speed-tie shuffles).
        battle = new Battle({ formatid: "gen1customgame", seed: seedWords });
        const realRandom = battle.prng.random.bind(battle.prng);
        battle.prng.random = (...args) => {
            const r = realRandom(...args);
            console.error(`  rng t${battle.turn} random(${args.join(",")}) = ${r}`);
            return r;
        };
        battle.setPlayer("p1", { name: "p1", team: P1 });
        battle.setPlayer("p2", { name: "p2", team: P2 });
        for (const [label, side] of [["p1", battle.sides[0]], ["p2", battle.sides[1]]]) {
            const p = side.active[0];
            console.error(`${label} ${p.set.species} L${p.level} maxhp=${p.maxhp} ability=${p.ability || "(none)"}`);
            console.error(`   storedStats = ${JSON.stringify(p.storedStats)}`);
            console.error(`   moveSlots = ${JSON.stringify((p.baseMoveSlots || []).map(s => ({ id: s.id, maxpp: s.maxpp })))}`);
        }
    } else {
        battle = new Battle({
            formatid: "gen1customgame", seed: seedWords,
            p1: { name: "p1", team: P1 }, p2: { name: "p2", team: P2 },
        });
    }
    // Normalize a fainted active to 'fnt' (Showdown's label for a 0-HP mon varies — 'fnt' when
    // it will be replaced, '' when it's the last mon and the battle ends). HP=0 is the real signal.
    const stat = (p) => (p.hp <= 0 ? "fnt" : code(p.status));
    const snap = () => {
        const a = battle.sides[0].active[0];
        const b = battle.sides[1].active[0];
        return {
            p1_sp: a.set.species, p1_hp: a.hp, p1_maxhp: a.maxhp, p1_status: stat(a),
            p2_sp: b.set.species, p2_hp: b.hp, p2_maxhp: b.maxhp, p2_status: stat(b),
        };
    };
    // "Attack with move 1; when forced to switch (active fainted), send in the lowest-index
    // alive teammate." The non-switching side passes ("") during a switch request.
    const firstSwitch = (side) => {
        for (let i = 0; i < side.pokemon.length; i++) {
            if (!side.pokemon[i].fainted && side.pokemon[i] !== side.active[0]) return i + 1;
        }
        return 0;
    };
    // Voluntary switches (CINNABAR_VOL=1): P1 switches on counter%5==1, P2 on %5==3, to the
    // lowest-index alive teammate. Keyed on a shared loop counter so both engines stay lockstep.
    const VOL = !!process.env.CINNABAR_VOL;
    const chooseFor = (side, si, c) => {
        const req = side.activeRequest;
        if (!req || req.wait) return "";
        if (req.forceSwitch && req.forceSwitch[0]) {
            const t = firstSwitch(side);
            return t ? `switch ${t}` : "pass";
        }
        if (VOL && ((si === 0 && c % 5 === 1) || (si === 1 && c % 5 === 3))) {
            const t = firstSwitch(side);
            if (t) return `switch ${t}`;
        }
        return "move 1";
    };
    const trace = [];
    let guard = 0, counter = 0;
    while (!battle.ended && guard++ < 2000) {
        battle.makeChoices(chooseFor(battle.sides[0], 0, counter), chooseFor(battle.sides[1], 1, counter));
        counter++;
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
