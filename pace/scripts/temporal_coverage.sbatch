#!/bin/bash
# Post-midterm section 4.1 — temporal clustering coverage.
#
# Re-runs curation over the sessions so out/<vid>/temporal_K<k>.{mp4,json} exists next to the
# kmeans/uniform outputs (run_session now writes all three), then regenerates the coverage
# table/plots WITH the Temporal method via hindsight.curation.analyze.
#
# Data: a balanced ~25-session subset (make_sessions --n N --per-person PER, ranked by evidence
# spans, <= PER per participant so LOPO still spans all 10). Each raw session mp4 is downloaded,
# embedded at 1 fps, curated (kmeans+temporal+uniform from the SAME 1 fps pool, so the coverage
# comparison is fair), and deleted (--delete-video unless KEEP_VIDEO=1): peak disk ~one video
# (~3-5 GB), not the full 228 GB. If cache/ already holds a session's .npz (same file), the embed
# is reused; a fresh download has a new mtime so CLIP re-embeds on CPU (slow but correct).
# The 1.34 GB curated sample (hf:hindsight-cs7641/hindsight-curated-frames) is NOT the input here
# — it is the VLM --curated-dir; temporal re-curates from raw frames so it can see short events.
#
# Submit FROM your work dir (usually via run_post_midterm.sh):
#   cd ~/scratch/hindsight && mkdir -p logs
#   sbatch "$HINDSIGHT_REPO/pace/scripts/temporal_coverage.sbatch"
# Settings: HINDSIGHT_REPO, HINDSIGHT_ENV (curation env), KS ["32 64 100 150"], SESSIONS
# (a sessions.txt; default = a fresh make_sessions --n N subset), N [25], PER [3],
# KEEP_VIDEO=1 (keep each mp4 after embedding, for sharing with the frame scorer; needs ~70 GB).
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

# Session list: a balanced evidence-ranked subset by default (shared with the frame scorer when
# run_post_midterm.sh passes SESSIONS). The temporal-done check is done below, not via
# make_sessions --skip-done (which only checks kmeans/uniform and would skip everything).
SESSIONS="${SESSIONS:-sessions_post_midterm.txt}"
N="${N:-25}"; PER="${PER:-3}"
if [ ! -s "$SESSIONS" ]; then
  python -m hindsight.curation.make_sessions --n "$N" --per-person "$PER" --out "$SESSIONS"
fi
DELETE="--delete-video"; [ -n "${KEEP_VIDEO:-}" ] && DELETE=""

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
    --ks $KS $DELETE < /dev/null || { echo "!! $vid failed"; FAILED=1; }
done < "$SESSIONS"

# Coverage table + plots, now including the Temporal row/line
mkdir -p results/post_midterm
python -m hindsight.curation.analyze --out out --results results/post_midterm/curation --k-bar 100 \
  || { echo "!! analyze failed"; FAILED=1; }

[ $FAILED -eq 0 ] || exit 1
echo "== finished at $(date): results/post_midterm/curation/coverage_summary.csv"
