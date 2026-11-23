#!/bin/bash
# Build (or repair) the CPU curation conda env the post-midterm jobs activate:
# ~/scratch/hindsight/env with torch (CPU), open_clip, cv2, sklearn, pandas, matplotlib and the
# hindsight package (editable). Idempotent: an existing env is reused and just re-verified.
#
#   export HINDSIGHT_REPO=~/scratch/hindsight/HINDSIGHT   # the rsync'd code (see docs/PACE.md 6)
#   bash "$HINDSIGHT_REPO/pace/scripts/setup_env.sh"
#
# Settings: HINDSIGHT_REPO [~/scratch/hindsight/HINDSIGHT], CURATION_ENV [~/scratch/hindsight/env],
# PYVER [3.11], TORCH_INDEX [https://download.pytorch.org/whl/cpu] (set to a cu12x URL for GPU torch).
# Optional HF_TOKEN (from the ENVIRONMENT, never hard-coded here): if set, it is written to the
# chmod-600 $HF_HOME/token file that every job reads, so no interactive `hf auth login` is needed.
#   HF_TOKEN=hf_xxx bash "$HINDSIGHT_REPO/pace/scripts/setup_env.sh"
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HINDSIGHT}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
PYVER="${PYVER:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"

[ -d "$REPO/server" ] || { echo "ERROR: no $REPO/server - rsync the code up first (docs/PACE.md section 6)" >&2; exit 1; }

if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p "$HF_HOME"
  umask 077; printf '%s' "$HF_TOKEN" > "$HF_HOME/token"; chmod 600 "$HF_HOME/token"
  echo "== wrote HF token to $HF_HOME/token (chmod 600); jobs read it via HF_HOME"
fi

module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$CURATION_ENV" ]; then
  echo "== creating $CURATION_ENV (python $PYVER)"
  conda create -y -p "$CURATION_ENV" "python=$PYVER"
else
  echo "== reusing existing env $CURATION_ENV"
fi
conda activate "$CURATION_ENV"

python -m pip install --upgrade pip
echo "== installing CPU torch from $TORCH_INDEX"
python -m pip install torch --index-url "$TORCH_INDEX"
echo "== installing hindsight[curation] (editable)"
python -m pip install -e "$REPO/server[curation]"

echo "== verifying imports"
PYTHONPATH="$REPO/server" python - <<'PY'
import sklearn, open_clip, cv2, pandas, matplotlib, numpy, scipy, torch
import hindsight.curation.frame_scorer, hindsight.curation.run_session
import hindsight.video.curation_temporal, hindsight.evaluation.stacking
print("OK: torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| sklearn", sklearn.__version__)
PY
echo "== done. activate with:  module load anaconda3 && conda activate $CURATION_ENV"
