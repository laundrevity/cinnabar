// Part A of the engine PoC: does @pkmn/engine build on this machine, and how many
// Gen 1 battles/sec can it play out? Random vs random, no logging (max speed).
// API mirrors the engine's own examples/js/example.ts (main branch).
//
//   cd poc/engine && npm install && node benchmark.mjs [N]
//
// Compare battles/sec to our Showdown self-play rate. A ~1000x gap is the whole
// reason we'd take on the integration.

import {Generations} from '@pkmn/data';
import {Dex} from '@pkmn/dex';
import {Battle, Choice} from '@pkmn/engine';
import {Team} from '@pkmn/sets';

// Two known-valid Gen 1 teams (packed format), from the engine's example.
const PACKED_1 =
  'Fushigidane|Bulbasaur||-|SleepPowder,SwordsDance,RazorLeaf,BodySlam|||||||]' +
  'Hitokage|Charmander||-|FireBlast,FireSpin,Slash,Counter|||||||]' +
  'Zenigame|Squirtle||-|Surf,Blizzard,BodySlam,Rest|||||||]' +
  'Pikachuu|Pikachu||-|Thunderbolt,ThunderWave,Surf,SeismicToss|||||||]' +
  'Koratta|Rattata||-|SuperFang,BodySlam,Blizzard,Thunderbolt|||||||]' +
  'Poppo|Pidgey||-|DoubleEdge,QuickAttack,WingAttack,MirrorMove|||||||';
const PACKED_2 =
  'Kentarosu|Tauros||-|BodySlam,HyperBeam,Blizzard,Earthquake|||||||]' +
  'Rakkii|Chansey||-|Reflect,SeismicToss,SoftBoiled,ThunderWave|||||||]' +
  'Kabigon|Snorlax||-|BodySlam,Reflect,Rest,IceBeam|||||||]' +
  'Nasshii|Exeggutor||-|SleepPowder,Psychic,Explosion,DoubleEdge|||||||]' +
  'Sutaamii|Starmie||-|Recover,ThunderWave,Blizzard,Thunderbolt|||||||]' +
  'Fuudin|Alakazam||-|Psychic,SeismicToss,ThunderWave,Recover|||||||';

const gen = new Generations(Dex).get(1);
const P1 = Team.unpack(PACKED_1, Dex).team;
const P2 = Team.unpack(PACKED_2, Dex).team;

const r = () => Math.floor(Math.random() * 256);
const randSeed = () => [r(), r(), r(), r()];

function playOne() {
  const battle = Battle.create(gen, {
    p1: {name: 'A', team: P1},
    p2: {name: 'B', team: P2},
    seed: randSeed(),
    showdown: true,
    log: false, // logging off for max throughput
  });
  let result, c1 = Choice.pass, c2 = Choice.pass;
  while (!(result = battle.update(c1, c2)).type) {
    const a = battle.choices('p1', result);
    const b = battle.choices('p2', result);
    c1 = a[Math.floor(Math.random() * a.length)];
    c2 = b[Math.floor(Math.random() * b.length)];
  }
  return battle.turn;
}

const N = Number(process.argv[2] || 20000);
for (let i = 0; i < 200; i++) playOne(); // warmup
const t0 = performance.now();
let turns = 0;
for (let i = 0; i < N; i++) turns += playOne();
const secs = (performance.now() - t0) / 1000;
console.log(
  `${N} battles in ${secs.toFixed(2)}s = ${Math.round(N / secs)} battles/sec ` +
  `(avg ${(turns / N).toFixed(1)} turns/battle)`
);
