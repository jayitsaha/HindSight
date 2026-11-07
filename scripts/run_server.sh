#!/usr/bin/env bash
# Start the HINDSIGHT API. Override host/port with HOST/PORT.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/uvicorn hindsight.api.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --reload
