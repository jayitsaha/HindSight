#!/bin/bash
# =====================================================================================================
# Whole SuperMemory-VQA Object & Location pipeline in one go: queue it and walk away.
#
#   cd ~/scratch/hindsight && bash "$HINDSIGHT_REPO/pace/scripts/run_full_pipeline.sh"
#
# Run on the LOGIN node from your work dir (the one holding out/ from earlier curation runs). It submits a
# chain of Slurm jobs, each starting when the previous step ends (afterany: a failed task does not block
# the rest; later steps use whatever finished):
#
#   1. curate   array job, one task per session not yet in out/ (KMeans + uniform for every K)
#   2. vlm      Gemma 4 over every question whose sessions are curated: curated KMeans, curated uniform,
#               blind (one job each, per K in VLM_KS)
#   3. finalize coverage tables/plots over all sessions + side-by-side VLM table -> results/full/SUMMARY.md
#
# Settings (environment variables, defaults in brackets):
#   HINDSIGHT_REPO  clone on the frame-curation branch   [~/scratch/hindsight/HindSight]
#   CURATION_ENV    conda env used for curation          [~/scratch/hindsight/env]
#   HINDSIGHT_ENV   Gemma 4 env (Transformers 5)         [~/scratch/hindsight/env-gemma4]
#   KS              K values to curate                   ["32 64 100 150"]
#   VLM_KS          K values to evaluate with Gemma      ["100"]
#   IMAGE_TOKENS    Gemma visual tokens per frame        [140]
#   MAX_PARALLEL    curation tasks at once               [10]
#   MODEL           VLM                                  [google/gemma-4-E4B-it]
#   DRY_RUN=1       print the sbatch commands, submit nothing
# Re-running the script is safe: finished sessions, K values and answered questions are skipped.
# =====================================================================================================
set -euo pipefail

WORK="$(pwd)"
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
CURATION_ENV="${CURATION_ENV:-$HOME/scratch/hindsight/env}"
VLM_ENV="${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env-gemma4}"
KS="${KS:-32 64 100 150}"
VLM_KS="${VLM_KS:-100}"
IMAGE_TOKENS="${IMAGE_TOKENS:-140}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
S="$REPO/pace/scripts"

die() { echo "ERROR: $*" >&2; exit 1; }
submit() {   # prints the job id; in DRY_RUN prints the command and a fake id
    if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

echo "== HindSight full pipeline | work dir $WORK | repo $REPO ($(git -C "$REPO" branch --show-current 2>/dev/null || echo '?'))"

# ---------------------------------------------------------------- pre-flight (fail now, not at 3 am)
[ -d "$REPO/server/hindsight/evaluation" ] || die "$REPO has no hindsight.evaluation: git checkout frame-curation && git pull"
[ -d "$CURATION_ENV" ] || die "curation env not found: $CURATION_ENV (set CURATION_ENV)"
[ -d "$VLM_ENV" ] || die "Gemma 4 env not found: $VLM_ENV (docs/PACE.md section 8, or set HINDSIGHT_ENV)"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$VLM_ENV"
python -c "import transformers, cv2; from transformers import AutoModelForMultimodalLM" 2>/dev/null \
    || die "$VLM_ENV cannot import transformers>=5 / AutoModelForMultimodalLM / cv2"
python - "$MODEL" <<'EOF' || die "model not in HF_HOME=$HF_HOME: run  hf download $MODEL  on the login node first"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True)
EOF
conda deactivate

conda activate "$CURATION_ENV"
python -c "import open_clip, cv2, pandas, matplotlib" 2>/dev/null || die "$CURATION_ENV is missing open_clip/cv2/pandas/matplotlib"
python -c "from hindsight.evaluation.run import default_qa_path; print('questions:', default_qa_path())" \
    || die "could not get all_qa.json (hf auth login?)"
mkdir -p logs out results/full
echo "== out/ already has $(find out -mindepth 1 -maxdepth 1 -type d | wc -l) session folders"

# ---------------------------------------------------------------- 1. curation of the remaining sessions
python -m hindsight.curation.make_sessions --all --skip-done out --ks $KS --out sessions_rest.txt
N=$(grep -c . sessions_rest.txt || true)
conda deactivate

DEP=""
if [ "$N" -gt 0 ]; then
    CUR_ID=$(submit --array=0-$((N - 1))%"$MAX_PARALLEL" \
        --export=ALL,SESSIONS=sessions_rest.txt,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV" \
        --job-name=hs-curate-rest "$S/curate_array.sbatch")
    # curate_array.sbatch runs --ks 32 64 100 150; KS only decides which sessions count as done
    DEP="--dependency=afterany:$CUR_ID"
    echo "== [1] curation: $N sessions -> job $CUR_ID"
else
    echo "== [1] curation: nothing left to do"
fi

# ---------------------------------------------------------------- 2. Gemma 4 on every curated question
# Each condition = a job + a continuation job that resumes from raw_predictions.jsonl if the first one hit the
# 8 h limit (when nothing is left it exits in about a minute without loading the model).
VLM_IDS=()
VLM_DIRS=()
vlm_job() {   # vlm_job <name> <output dir> <run.py args...>
    local name=$1 dir=$2; shift 2
    local a b
    a=$(submit $DEP --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV" --job-name="$name" \
        "$S/vlm_eval.sbatch" --model-name "$MODEL" --output-dir "$dir" "$@")
    b=$(submit --dependency=afterany:"$a" --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$VLM_ENV" \
        --job-name="$name-cont" "$S/vlm_eval.sbatch" --model-name "$MODEL" --output-dir "$dir" "$@")
    VLM_IDS+=("$b"); VLM_DIRS+=("$dir")
    echo "== [2] $name -> jobs $a (+ continuation $b)"
}
for K in $VLM_KS; do
    for M in kmeans uniform; do
        vlm_job "hs-vlm-$M-K$K" "results/full/vlm/curated_${M}_K${K}" \
            --condition curated --curated-dir out --method "$M" --k "$K" --image-tokens "$IMAGE_TOKENS"
    done
done
FIRST_K=${VLM_KS%% *}
vlm_job hs-vlm-blind results/full/vlm/blind --condition blind --curated-dir out --method kmeans --k "$FIRST_K"

# ---------------------------------------------------------------- 3. finalize
FIN_ID=$(submit --dependency=afterany:$(IFS=:; echo "${VLM_IDS[*]}") \
    --export=ALL,HINDSIGHT_REPO="$REPO",HINDSIGHT_ENV="$CURATION_ENV",VLM_DIRS="${VLM_DIRS[*]}",KS="$KS" \
    "$S/finalize_full.sbatch")
echo "== [3] finalize -> job $FIN_ID"

{
    echo "submitted $(date)"; echo "curate ${CUR_ID:-none} ($N sessions)"; echo "vlm ${VLM_IDS[*]}"; echo "finalize $FIN_ID"
} > results/full/jobs.txt
cat <<EOF

All queued (ids in results/full/jobs.txt). Check with:  squeue --me
In the morning:  cat results/full/SUMMARY.md
Logs: logs/hs-curate-rest-*  logs/hs-vlm-*  logs/hs-final-*
If something died (time limit, node failure), just run this script again: finished work is skipped.
EOF
