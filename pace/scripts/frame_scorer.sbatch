#!/bin/bash
# Post-midterm section 5.1 — supervised evidence-relevance frame scorer, leave-one-participant-out eval.
#
# DEPENDS ON THE CURATION EMBEDDING CACHE: `frame_scorer eval` scores the real sessions' CLIP
# frame embeddings, which it reads from the cache/ dir written by curate_array.sbatch /
# run_full_pipeline.sh. Run curation first. On a cache miss it re-downloads the mp4 from
# Hugging Face and re-embeds with CLIP on the CPU (slow); only the query-text embeddings are
# always computed here (cheap on CPU).
#
# `eval` prints the per-participant AP table (base rate vs cosine vs scorer) to stdout — there
# is no --out flag — so stdout is tee'd into results/post_midterm/frame_scorer/.
#
# Submit FROM the work dir that holds cache/ (usually via run_post_midterm.sh):
#   cd ~/scratch/hindsight && mkdir -p logs
#   sbatch "$HINDSIGHT_REPO/pace/scripts/frame_scorer.sbatch"
# Settings: HINDSIGHT_REPO, HINDSIGHT_ENV (curation env), SCORER_MODEL [logreg|mlp|gbm].
#
# CPU job: no --gres (PACE ICE CPU nodes need none). If your queue requires a partition, use
# the same fallback as abstention.sbatch: -p ice-gpu --gres=gpu:l40s:1.
#SBATCH -J hs-scorer
#SBATCH -N1 --ntasks-per-node=4 --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${SCORER_MODEL:-logreg}"
OUT=results/post_midterm/frame_scorer
mkdir -p "$OUT"

[ -d cache ] || echo "WARNING: no cache/ in $(pwd) - every frame will be re-embedded on CPU (run curation first)"

echo "== $(hostname) at $(date): frame scorer LOPO eval (model=$MODEL)"
python -m hindsight.curation.frame_scorer eval --cache cache --data data --model "$MODEL" \
  | tee "$OUT/eval_${MODEL}.txt"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" -eq 0 ] || exit "$STATUS"
echo "== finished at $(date): $OUT/eval_${MODEL}.txt"
