#!/bin/bash
# Post-midterm section 5.1 — supervised evidence-relevance frame scorer, leave-one-participant-out eval.
#
# Data: the SAME balanced ~25-session subset as temporal_coverage.sbatch (SESSIONS, shared by
# run_post_midterm.sh). --sessions restricts the candidate pool to those video_ids so only ~25
# sessions are touched, not all ~69. Each session's raw mp4 is downloaded, embedded at 1 fps and
# deleted (--delete-video unless KEEP_VIDEO=1): peak disk ~one video (~3-5 GB). If cache/ already
# holds a session's .npz (same file), the embed is reused; a fresh download re-embeds on CPU.
# A warm cache from temporal_coverage.sbatch is only reused if KEEP_VIDEO=1 kept the mp4s (same
# mtime); otherwise this job re-downloads+re-embeds the subset itself.
#
# `eval` prints the per-participant AP table (base rate vs cosine vs scorer) to stdout — there
# is no --out flag — so stdout is tee'd into results/post_midterm/frame_scorer/.
#
# Submit FROM your work dir (usually via run_post_midterm.sh):
#   cd ~/scratch/hindsight && mkdir -p logs
#   sbatch "$HINDSIGHT_REPO/pace/scripts/frame_scorer.sbatch"
# Settings: HINDSIGHT_REPO, HINDSIGHT_ENV (curation env), SCORER_MODEL [logreg|mlp|gbm],
# SESSIONS [sessions_post_midterm.txt], N [25], PER [3], KEEP_VIDEO=1 (keep mp4s after embedding).
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

SESSIONS="${SESSIONS:-sessions_post_midterm.txt}"
N="${N:-25}"; PER="${PER:-3}"
if [ ! -s "$SESSIONS" ]; then
  python -m hindsight.curation.make_sessions --n "$N" --per-person "$PER" --out "$SESSIONS"
fi
DELETE="--delete-video"; [ -n "${KEEP_VIDEO:-}" ] && DELETE=""

echo "== $(hostname) at $(date): frame scorer LOPO eval (model=$MODEL, sessions=$SESSIONS)"
# --sessions restricts to the subset; --delete-video keeps peak disk at ~one video (~3-5 GB).
python -m hindsight.curation.frame_scorer eval --cache cache --data data --model "$MODEL" \
  --sessions "$SESSIONS" $DELETE \
  | tee "$OUT/eval_${MODEL}.txt"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" -eq 0 ] || exit "$STATUS"
echo "== finished at $(date): $OUT/eval_${MODEL}.txt"
