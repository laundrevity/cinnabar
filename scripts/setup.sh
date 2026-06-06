#!/usr/bin/env bash
#
# One-time setup: pull in the Pokémon Showdown submodule and build it.
# Run this from anywhere — it resolves paths relative to the repo root.
#
# Requires network access to github.com and the npm registry, so run it on
# your own machine (not in a locked-down sandbox).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBMODULE_PATH="server/pokemon-showdown"
SUBMODULE_URL="https://github.com/smogon/pokemon-showdown"

echo "==> Ensuring the Pokémon Showdown submodule is present"
if [ ! -f "$SUBMODULE_PATH/package.json" ]; then
  if grep -q "$SUBMODULE_PATH" .gitmodules 2>/dev/null; then
    # .gitmodules already declares it (e.g. fresh clone of this repo)
    git submodule update --init --recursive
  else
    # First time: register it as a submodule, pinned to the current master
    git submodule add "$SUBMODULE_URL" "$SUBMODULE_PATH"
  fi
else
  echo "    already present, skipping clone"
fi

echo "==> Installing Showdown dependencies (this can take a couple of minutes)"
cd "$SUBMODULE_PATH"
npm install

echo "==> Building Showdown"
node pokemon-showdown build || true

echo ""
echo "==> Done. Start a local server for training/play with:"
echo "      scripts/run-server.sh"
echo ""
echo "    Then point poke-env / your agent at ws://localhost:8000/showdown/websocket"
