#!/bin/bash
# ==============================================================================
# Hindsight: submit the SuperMemory-VQA pilot (oracle-clip + text-only control)
# on PACE ICE. The job itself downloads the needed session videos (one at a time,
# deleted after frames are cut), then evaluates both conditions.
#
#   HINDSIGHT_ENV=$SCRATCH_PROJECT/env-gemma4 bash scripts/submit_pilot.sh [model]
# Re-submitting resumes: frames already cut are skipped and finished questions are
# read from raw_predictions.jsonl. Env: SUBJECT (S1), N_QUESTIONS (40), PILOT_DIR.
# ==============================================================================

MODEL_NAME="${1:-google/gemma-4-E4B-it}"
TARGET_SCRATCH="${SCRATCH_PROJECT:-/home/hice1/vchheda7/scratch/cs7641project}"
ENV_PATH="${HINDSIGHT_ENV:-$TARGET_SCRATCH/env-gemma4}"
PILOT_DIR="${PILOT_DIR:-$TARGET_SCRATCH/pilot}"
TAG=$(basename "$MODEL_NAME")

SUBJECT="${SUBJECT:-S1}"
N_QUESTIONS="${N_QUESTIONS:-40}"
mkdir -p "$TARGET_SCRATCH/logs"

# Explicit NVIDIA GRES (a generic gpu:1 can land on AMD nodes); first of H200/H100.
AVAILABLE_GPU_GRES=$(sinfo -p ice-gpu -N -h -o "%G" 2>/dev/null || true)
GPU_TYPE=""
for candidate in h200 h100 a100; do
    if printf '%s\n' "$AVAILABLE_GPU_GRES" | grep -Eiq "gpu:${candidate}:"; then GPU_TYPE="$candidate"; break; fi
done
if [ -z "$GPU_TYPE" ]; then echo "ERROR: no NVIDIA GPU GRES on ice-gpu" >&2; exit 1; fi

JOB_FILE="$TARGET_SCRATCH/run_pilot.slurm"
cat > "$JOB_FILE" << EOF
#!/bin/bash
#SBATCH --job-name=hindsight-pilot
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=${TARGET_SCRATCH}/logs/pilot_%j.out
#SBATCH --error=${TARGET_SCRATCH}/logs/pilot_%j.err

module purge 2>/dev/null || true
module load anaconda3 2>/dev/null || true
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="${TARGET_SCRATCH}/hf_cache"
cd "${TARGET_SCRATCH}"

# Compute nodes need internet for this step; if the job log shows a download error,
# run the same command on the login node and resubmit.
python3 scripts/prepare_pilot.py --subject ${SUBJECT} --n ${N_QUESTIONS} --pilot-dir "${PILOT_DIR}" || exit 1
df -h "${TARGET_SCRATCH}" | tail -1

for COND in oracle blind; do
    python3 scripts/run_evaluation.py --backend hf --model-name "${MODEL_NAME}" \\
        --pilot-dir "${PILOT_DIR}" --condition \$COND \\
        --output-dir "${TARGET_SCRATCH}/results/pilot_${TAG}_\$COND"
done
python3 scripts/compare_conditions.py "${TARGET_SCRATCH}/results/pilot_${TAG}_oracle" "${TARGET_SCRATCH}/results/pilot_${TAG}_blind"
EOF
echo "GPU: $GPU_TYPE | model: $MODEL_NAME | pilot: $PILOT_DIR | $SUBJECT x $N_QUESTIONS"
sbatch "$JOB_FILE"
