#!/bin/bash
# Learned retriever (supervised): CPU only. Needs cache/ and out/ from the full run, and reuses the text
# embeddings that grounded.sbatch saved (results/full/grounded/text_emb.npz; encoded here if missing).
#
# Submit FROM the work dir of the full run:
#   cd ~/scratch/hindsight && mkdir -p logs && sbatch "$HINDSIGHT_REPO/pace/scripts/retriever.sbatch"
# Watch:   tail -f logs/hs-retriever-<jobid>.out
# Result:  results/full/retriever/RETRIEVER.md
# Extra arguments are passed through, e.g.  sbatch .../retriever.sbatch --sessions prior
#SBATCH -J hs-retriever
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=4 --mem=24G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REPO="${HINDSIGHT_REPO:-$HOME/scratch/hindsight/HindSight}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HINDSIGHT_ENV:-$HOME/scratch/hindsight/env}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hindsight/hf}"
export PYTHONPATH="$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "== $(hostname) at $(date), ${SLURM_CPUS_PER_TASK:-?} CPUs"
python -m hindsight.curation.learned_retriever --cache cache --out out --results results/full/retriever \
    --text-cache results/full/grounded/text_emb.npz --device cpu "$@"
echo "== finished at $(date)"
