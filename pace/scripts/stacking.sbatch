#!/bin/bash
# Post-midterm section 5.3 — cross-view stacking over the three Gemma log-prob views.
#
# DEPENDS ON THE _lp RUNS: it reads raw_predictions.jsonl (with option_logprobs) from
# results/full/vlm/{curated_kmeans_K100_lp,curated_uniform_K100_lp,blind_lp}, which are written
# by submit_abstention.sh's Gemma re-runs (GPU, docs/PACE.md section 10). If those dirs are
# missing, run  bash "$HINDSIGHT_REPO/pace/scripts/submit_abstention.sh"  first.
# The stacking step itself is pure scikit-learn — CPU, a few minutes.
#
# Submit FROM the work dir that holds results/full/vlm/ (usually via run_post_midterm.sh):
#   cd ~/scratch/hindsight && mkdir -p logs
#   sbatch "$HINDSIGHT_REPO/pace/scripts/stacking.sbatch"
# Extra arguments are passed to `python -m hindsight.evaluation.stacking`; with none, the
# default K=100 _lp dirs and --out results/post_midterm/stacking are used.
#
# CPU job: no --gres (PACE ICE CPU nodes need none). If your queue requires a partition, use
# the same fallback as abstention.sbatch: -p ice-gpu --gres=gpu:l40s:1.
#SBATCH -J hs-stack
#SBATCH -N1 --ntasks-per-node=4 --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"

if [ $# -eq 0 ]; then
  set -- results/full/vlm/curated_kmeans_K100_lp results/full/vlm/curated_uniform_K100_lp \
         results/full/vlm/blind_lp --out results/post_midterm/stacking
fi
for d in "$1" "$2" "$3"; do
  [ -f "$d/raw_predictions.jsonl" ] || {
    echo "ERROR: $d/raw_predictions.jsonl missing - run submit_abstention.sh (Gemma _lp runs) first" >&2
    exit 1
  }
done

echo "== $(hostname) at $(date): stacking $*"
python -m hindsight.evaluation.stacking "$@"
echo "== finished at $(date): see STACKING.md in the --out dir"
