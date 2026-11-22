#!/bin/bash
# Post-midterm section 4.1 — temporal clustering coverage.
#
# Re-runs curation over the sessions so out/<vid>/temporal_K<k>.{mp4,json} exists next to the
# kmeans/uniform outputs (run_session now writes all three), then regenerates the coverage
# table/plots WITH the Temporal method via hindsight.curation.analyze.
#
# CPU only: the CLIP embeddings from the original curation runs are reused from cache/, so
# re-curating a session is just clustering + a video write pass (plus re-downloading the mp4,
# since curate_array ran with --delete-video). If cache/ is cold for a session, CLIP falls back
# to the CPU (slow) — run the GPU curate_array.sbatch for that session instead.
#
# Submit FROM the work dir that holds out/ and cache/ (usually via run_post_midterm.sh):
#   cd ~/scratch/hindsight && mkdir -p logs
#   sbatch "$HINDSIGHT_REPO/pace/scripts/temporal_coverage.sbatch"
# Settings: HINDSIGHT_REPO, HINDSIGHT_ENV (curation env), KS ["32 64 100 150"], SESSIONS
# (a sessions.txt; default = every session, from make_sessions --all).
#
# CPU job: no --gres (PACE ICE CPU nodes need none). If your queue requires a partition, use
# the same fallback as abstention.sbatch: -p ice-gpu --gres=gpu:l40s:1.
#SBATCH -J hs-temporal
#SBATCH -N1 --ntasks-per-node=4 --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
KS="${KS:-32 64 100 150}"

echo "== $(hostname) at $(date): temporal re-curation + coverage (K = $KS)"

# Session list: every session by default. NOTE: make_sessions --skip-done only checks
# kmeans/uniform, so it would skip everything — the temporal-done check is done here instead.
SESSIONS="${SESSIONS:-sessions_temporal.txt}"
if [ ! -s "$SESSIONS" ]; then
  python -m hindsight.curation.make_sessions --all --out "$SESSIONS"
fi

FAILED=0
i=-1
while IFS=$'\t' read -r repo_path vid _; do
  [ -n "$vid" ] || continue      # run_session indexes non-blank lines only
  i=$((i + 1))
  DONE=1
  for k in $KS; do [ -f "out/$vid/temporal_K$k.json" ] || DONE=0; done
  if [ "$DONE" -eq 1 ]; then echo "-- line $i $vid: temporal already done, skipping"; continue; fi
  echo "-- line $i $vid"
  # run_session skips each K whose kmeans+uniform+temporal manifests all exist already
  python -m hindsight.curation.run_session --index "$i" --sessions "$SESSIONS" \
    --ks $KS --delete-video < /dev/null || { echo "!! $vid failed"; FAILED=1; }
done < "$SESSIONS"

# Coverage table + plots, now including the Temporal row/line
mkdir -p results/post_midterm
python -m hindsight.curation.analyze --out out --results results/post_midterm/curation --k-bar 100 \
  || { echo "!! analyze failed"; FAILED=1; }

[ $FAILED -eq 0 ] || exit 1
echo "== finished at $(date): results/post_midterm/curation/coverage_summary.csv"
