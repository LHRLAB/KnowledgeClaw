#!/usr/bin/env bash
# Start the KnowledgeClaw web server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${KNOWLEDGECLAW_HOST:-0.0.0.0}"
PORT="${KNOWLEDGECLAW_PORT:-8000}"
RELOAD="${KNOWLEDGECLAW_RELOAD:-false}"

echo "Starting KnowledgeClaw on http://${HOST}:${PORT}"

if [ "$RELOAD" = "true" ]; then
  uvicorn app:app --host "$HOST" --port "$PORT" --reload
else
  uvicorn app:app --host "$HOST" --port "$PORT"
fi
