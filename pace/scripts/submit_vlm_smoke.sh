#!/bin/bash
# ==============================================================================
# Hindsight VLM smoke test on PACE ICE (synthetic video -> VLM answer). NVIDIA GPUs only.
#   HINDSIGHT_ENV=<env> bash pace/scripts/submit_vlm_smoke.sh [model]
# Env: SCRATCH_PROJECT (work dir, default ~/scratch/hindsight), HINDSIGHT_REPO (clone, default $SCRATCH_PROJECT/HindSight)
# Detects NVIDIA GRES on `ice-gpu` and never submits a generic GPU request.
# Defaults to HuggingFaceTB/SmolVLM-Instruct; supports recent Gemma 4 via an
# independently-created environment selected with HINDSIGHT_ENV.
# ==============================================================================

# Model choice: default to SmolVLM-Instruct (2.2B)
MODEL_NAME="${1:-HuggingFaceTB/SmolVLM-Instruct}"
MODEL_NAME_LOWER=$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_NAME_LOWER" == *gemma-4-e2b* || "$MODEL_NAME_LOWER" == *gemma-4-e4b* ]]; then
    MEM_REQ="32G"
    TIME_REQ="01:00:00"
elif [[ "$MODEL_NAME_LOWER" == *gemma-4* || "$MODEL_NAME_LOWER" == *gemma4* ]]; then
    MEM_REQ="64G"
    TIME_REQ="02:00:00"
else
    MEM_REQ="32G"
    TIME_REQ="00:30:00"
fi

echo "======================================================================"
echo "        PACE ICE BATCH SUBMISSION (NVIDIA GPU ONLY)                   "
echo "  Target Model: $MODEL_NAME"
echo "  Partition:    ice-gpu"
echo "======================================================================"

# 1. Target PACE ICE Scratch Directory
TARGET_SCRATCH="${SCRATCH_PROJECT:-$HOME/scratch/hindsight}"
REPO="${HINDSIGHT_REPO:-$TARGET_SCRATCH/HindSight}"
mkdir -p "$TARGET_SCRATCH/logs"
mkdir -p "$TARGET_SCRATCH/hf_cache"

ENV_PATH="${HINDSIGHT_ENV:-$TARGET_SCRATCH/env-vlm}"

# Choose only a GPU type explicitly reported by this partition. A generic
# --gres=gpu:1 can select AMD MI210 nodes, which cannot use the CUDA wheel.
AVAILABLE_GPU_GRES=$(sinfo -p ice-gpu -N -h -o "%G" 2>/dev/null || true)
GPU_TYPE=""
for candidate in h200 h100 a100 a40 rtx_6000 rtx6000; do
    if printf '%s\n' "$AVAILABLE_GPU_GRES" | grep -Eiq "gpu:${candidate}:"; then
        GPU_TYPE="$candidate"
        break
    fi
done

if [ -z "$GPU_TYPE" ]; then
    echo "ERROR: No NVIDIA GPU GRES found on partition ice-gpu; job was not submitted." >&2
    echo "Reported GRES: ${AVAILABLE_GPU_GRES:-none}"
    exit 1
fi
GPU_GRES="gpu:${GPU_TYPE}:1"

echo "[1/3] Scratch Directory: $TARGET_SCRATCH"
echo "[1/3] HuggingFace Cache: $TARGET_SCRATCH/hf_cache"
echo "[1/3] Conda Environment: $ENV_PATH"
echo "[1/3] NVIDIA GPU Request: $GPU_GRES"

# 2. Generate Slurm Job Script
JOB_FILE="$TARGET_SCRATCH/run_sample_video.slurm"
cat << 'EOF' > "$JOB_FILE"
#!/bin/bash
#SBATCH --job-name=hindsight-ice-sample
#SBATCH --partition=ice-gpu
#SBATCH --gres=__GPU_GRES__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=__MEM_REQ__
#SBATCH --time=__TIME_REQ__
#SBATCH --output=__SCRATCH__/logs/sample_%j.out
#SBATCH --error=__SCRATCH__/logs/sample_%j.err

ENV_PATH="__ENV_PATH__"

echo "=== PACE ICE Job Started: $(date) ==="
echo "Node: $(hostname)"

# Load Cluster Modules
module purge 2>/dev/null || true
module load anaconda3 2>/dev/null || true
if command -v nvidia-smi >/dev/null 2>&1; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Activate the project environment explicitly; never fall back to base Anaconda.
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -d "$ENV_PATH" ]; then
    echo "ERROR: Conda environment not found: $ENV_PATH" >&2
    exit 1
fi
conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1

echo "Python: $(which python3)"
if [[ "__MODEL_NAME__" == *gemma-4* || "__MODEL_NAME__" == *gemma4* ]]; then
    python3 -c 'import sys, torch; from transformers import AutoProcessor, AutoModelForMultimodalLM; print("Python executable:", sys.executable); print("PyTorch:", torch.__version__); print("Transformers Gemma 4 imports: OK"); print("GPU available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
else
    python3 -c 'import sys, torch; from transformers import AutoProcessor; print("Python executable:", sys.executable); print("PyTorch:", torch.__version__); from transformers import AutoModelForImageTextToText; print("Transformers imports: OK (v5 API)"); print("GPU available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' 2>/dev/null || \
    python3 -c 'import sys, torch; from transformers import AutoProcessor, AutoModelForVision2Seq; print("Python executable:", sys.executable); print("PyTorch:", torch.__version__); print("Transformers imports: OK (v4 API)"); print("GPU available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
fi

# Inspect GPU (NVIDIA or AMD)
if command -v nvidia-smi &> /dev/null; then
    echo "--- NVIDIA GPU Detected ---"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
elif command -v rocm-smi &> /dev/null; then
    echo "--- AMD GPU Detected ---"
    rocm-smi || true
else
    echo "--- No SMI utility found; checking PyTorch directly ---"
fi

# Scratch & Hugging Face Cache Setup
export TARGET_SCRATCH="__SCRATCH__"
export HF_HOME="$TARGET_SCRATCH/hf_cache"
export SCRATCH_PROJECT="$TARGET_SCRATCH"

cd "$TARGET_SCRATCH"

# Run end-to-end sample video inference
export PYTHONPATH="__REPO__/server"
python3 -m hindsight.vlm.smoke_test --model-name "__MODEL_NAME__"

echo "=== Job Finished: $(date) ==="
EOF

# Substitute target model
sed -i "s|__MODEL_NAME__|$MODEL_NAME|g" "$JOB_FILE"
sed -i "s|__ENV_PATH__|$ENV_PATH|g" "$JOB_FILE"
sed -i "s|__GPU_GRES__|$GPU_GRES|g" "$JOB_FILE"
sed -i "s|__MEM_REQ__|$MEM_REQ|g" "$JOB_FILE"
sed -i "s|__TIME_REQ__|$TIME_REQ|g" "$JOB_FILE"
sed -i "s|__SCRATCH__|$TARGET_SCRATCH|g" "$JOB_FILE"
sed -i "s|__REPO__|$REPO|g" "$JOB_FILE"
chmod +x "$JOB_FILE"

# 3. Submit Job via sbatch
echo -e "\n[2/3] Generated Slurm script at: $JOB_FILE"
echo -e "[3/3] Submitting job via sbatch..."

SUBMIT_OUTPUT=$(sbatch "$JOB_FILE")
echo "$SUBMIT_OUTPUT"

JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')

if [ -n "$JOB_ID" ]; then
    echo "======================================================================"
    echo "Job successfully submitted to PACE ICE! (Job ID: $JOB_ID)"
    echo ""
    echo "1. Check queue status:"
    echo "   squeue -u $USER"
    echo ""
    echo "2. Stream output logs:"
    echo "   tail -f $TARGET_SCRATCH/logs/sample_${JOB_ID}.out"
    echo ""
    echo "3. Stream error logs (if any):"
    echo "   tail -f $TARGET_SCRATCH/logs/sample_${JOB_ID}.err"
    echo "======================================================================"
fi
