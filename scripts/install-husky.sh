#!/usr/bin/env sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Honor common opt-out flags
if [ "${HUSKY:-}" = "0" ] || [ "${HUSKY_SKIP_INSTALL:-}" = "1" ]; then
  echo "Husky install skipped (disabled by environment)"
  exit 0
fi

# Skip in environments without Git metadata
if [ ! -d .git ]; then
  echo "Husky install skipped (no .git directory)"
  exit 0
fi

# Skip if the Husky binary is unavailable (e.g., offline installs)
if [ ! -x ./node_modules/.bin/husky ]; then
  echo "Husky binary not found; skipping hook installation"
  exit 0
fi

./node_modules/.bin/husky install
