#!/usr/bin/env bash
# Push the repo to PACE scratch. The laptop is the source of truth; PACE only
# updates when you run this. Override the target with PACE_REMOTE / PACE_DEST.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${PACE_REMOTE:-jsaha34@login-ice.pace.gatech.edu}"
DEST="${PACE_DEST:-\$HOME/scratch/hindsight/HINDSIGHT/}"

ssh "$REMOTE" "mkdir -p $DEST"
rsync -avP \
  --exclude '.venv' --exclude '.git' --exclude 'node_modules' \
  --exclude 'server/data' --exclude 'server/artifacts' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.expo' \
  ./ "$REMOTE:$DEST"
echo "pushed to $REMOTE:$DEST"
