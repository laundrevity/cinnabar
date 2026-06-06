# server/

Pokémon Showdown — the game engine and interface for this project.

`pokemon-showdown/` is a **git submodule** pinned to a specific commit of
[smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown). It isn't checked in
directly; populate it with:

```bash
scripts/setup.sh        # adds the submodule, installs deps, builds
```

Then run a local server:

```bash
scripts/run-server.sh   # http://localhost:8000, --no-security
```

Notes:

- `--no-security` removes logins, rate limits, and chat filters — necessary for fast local
  training, but **never expose such a server to the internet.**
- The WebSocket endpoint your agent connects to is
  `ws://localhost:8000/showdown/websocket`.
- Bump the submodule deliberately (`cd pokemon-showdown && git checkout <commit>`, then
  commit the new pointer) rather than letting it drift.
