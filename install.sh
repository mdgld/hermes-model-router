#!/usr/bin/env bash
# Thin wrapper around the Python installer so the plugin can do profile-aware,
# idempotent setup without brittle shell regex hacks.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_PYTHON="${HOME}/.hermes/hermes-agent/venv/bin/python3"
if [ -x "$HERMES_PYTHON" ] && "$HERMES_PYTHON" -c "import yaml" 2>/dev/null; then
    exec "$HERMES_PYTHON" "$PLUGIN_DIR/install.py" "$@"
fi
exec python3 "$PLUGIN_DIR/install.py" "$@"
