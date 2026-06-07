# Engine PoC — is @pkmn/engine viable for us *now*?

De-risking the engine integration before committing weeks. See
`../../docs/engine-integration.md` for the full plan. Two questions, in order.

## Part A — does the engine build, and how fast is it? (JS)

```sh
cd poc/engine
npm install        # postinstall fetches a Zig compiler and builds the native addon
node benchmark.mjs            # 20,000 random Gen 1 battles
node benchmark.mjs 100000     # more samples
```

Expect output like `20000 battles in X.XXs = NNNN battles/sec`. Compare that
battles/sec to our Showdown self-play rate (tens of battles per multi-second
iteration). If it's hundreds-of-thousands to millions/sec, the throughput thesis
holds and the integration is worth it.

If `npm install` fails: it's pre-v0.1 and tracks Zig master, so a toolchain
mismatch is the likely culprit — note the error; that itself is a useful result.

## Part B — can *Python* drive it? (the real question)

JS speed is necessary but not sufficient — our training loop is Python. Build the
community binding [PyKMN](https://github.com/AnnikaCodes/PyKMN) from source:

```sh
python3 -m pip install ruff mypy build coverage cffi requests types-cffi types-requests types-setuptools
git clone https://github.com/AnnikaCodes/PyKMN && cd PyKMN
./build.sh python3          # or: python3 -m build && pip install --find-links=dist pykmn
python3 -m unittest discover tests   # confirm it actually works against the current engine
```

Docs/examples: <https://annikacodes.github.io/PyKMN/latest/> and the repo's
`examples/`. The risk: PyKMN is third-party and may lag the engine's breaking
changes. If it imports and steps a Gen 1 battle, we're in business; if it's stale,
the fallback is our own ctypes bindings against `libpkmn` (more work).

## What to report back

1. Did Part A build, and the battles/sec number.
2. Did PyKMN install + pass its tests (can Python step a battle).

Those two answers decide whether the engine path is real today or blocked on v0.1.
