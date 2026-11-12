#!/bin/bash
#SBATCH --job-name=hindsight-vlm
#SBATCH --output=results/pace_logs_%j.out
#SBATCH --error=results/pace_logs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# ==============================================================================
# Georgia Tech PACE Slurm Submission Script for Hindsight (CS 7641)
# ==============================================================================

echo "=== Job Started: $(date) ==="
echo "Node: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"

# 1. Load Modules (Adjust according to your PACE cluster environment, e.g. Phoenix)
module purge
module load anaconda3
module load cuda/12.1

# 2. Activate Conda Environment
# conda activate hindsight-env

# 3. PACE ICE Scratch Directory Setup
# Uses /home/hice1/vchheda7/scratch/cs7641project for caches and heavy outputs
if [ -n "$SCRATCH_PROJECT" ]; then
    TARGET_SCRATCH="$SCRATCH_PROJECT"
elif [ -d "/home/hice1/$USER/scratch" ]; then
    TARGET_SCRATCH="/home/hice1/$USER/scratch/cs7641project"
elif [ -d "$HOME/scratch" ]; then
    TARGET_SCRATCH="$HOME/scratch/cs7641project"
else
    TARGET_SCRATCH="/home/hice1/vchheda7/scratch/cs7641project"
fi

mkdir -p "$TARGET_SCRATCH/hf_cache"
export HF_HOME="$TARGET_SCRATCH/hf_cache"
echo "Project Scratch Directory: $TARGET_SCRATCH"
echo "HuggingFace Cache directory set to: $HF_HOME"

# 4. Run Evaluation on GPU
cd $TARGET_SCRATCH

# Example: Run Pixtral-12B (Mistral AI, France) on H100 / H200 in bfloat16
python3 scripts/run_evaluation.py \
    --backend hf \
    --model-name "mistralai/Pixtral-12B-2409" \
    --task-category "Object and Location Memory" \
    --output-dir "$SCRATCH_PROJECT/results/pace_eval_${SLURM_JOB_ID}"

echo "=== Job Finished: $(date) ==="
