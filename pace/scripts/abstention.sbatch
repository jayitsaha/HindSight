#!/bin/bash
# Abstention classifier (scikit-learn, leave-one-participant-out) on the Gemma runs that recorded option log-probs.
# Submitted by submit_abstention.sh; arguments go to `python -m hindsight.evaluation.abstention`.
# CPU only (about a minute); on ice-gpu only because that is where ICE jobs go. Can also be run on the login node:
#   PYTHONPATH=$HINDSIGHT_REPO/server python -m hindsight.evaluation.abstention results/full/vlm/*_lp --out results/full/abstention
#SBATCH -J hs-abstain
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N1 --ntasks-per-node=4 --mem=16G
#SBATCH -t 0:45:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
echo "== $(hostname) at $(date): $*"
python -m hindsight.evaluation.abstention "$@"
echo "== finished at $(date)"
