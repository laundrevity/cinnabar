#!/usr/bin/env bash
#
# Start a local Pokémon Showdown server for self-play / training / browser play.
#
# --no-security disables logins, rate limits, and chat filters. This is what you
# want for local training (poke-env spins up many connections fast). NEVER expose
# a --no-security server to the public internet.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN="$ROOT/server/pokemon-showdown"

if [ ! -f "$SHOWDOWN/package.json" ]; then
  echo "Showdown isn't set up yet. Run scripts/setup.sh first." >&2
  exit 1
fi

cd "$SHOWDOWN"
PORT="${1:-8000}"
echo "Starting Pokémon Showdown on http://localhost:$PORT  (Ctrl-C to stop)"
exec node pokemon-showdown start "$PORT" --no-security
