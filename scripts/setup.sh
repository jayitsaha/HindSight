#!/usr/bin/env bash
# One-shot setup: venv, deps, synthetic data. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3.14}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -e server
# optional local CLIP fallback (skip with NO_TORCH=1)
if [ "${NO_TORCH:-0}" != "1" ]; then
  ./.venv/bin/python -m pip install -q torch open-clip-torch || echo "torch install skipped"
fi

./.venv/bin/python -m hindsight.cli gen
./.venv/bin/python -m hindsight.cli status
echo "setup done. train/eval: ./.venv/bin/python -m hindsight.training.eval_all"
