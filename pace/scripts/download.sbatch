#!/bin/bash
#SBATCH -J hs-download
#SBATCH -N1 --ntasks-per-node=4 --mem=32G
#SBATCH -t8:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# Pull Ego4D annotations + omnivore features to scratch. CPU only (no GPU).
# The ego4d CLI resumes partial downloads, so resubmit if the 8h limit is hit.
# AWS creds must be configured first (aws configure) from your license approval.
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/hindsight/envs/hs
export EGO4D_ROOT=~/scratch/hindsight/ego4d
mkdir -p "$EGO4D_ROOT"

# annotations for all four benchmarks (~3 GB)
ego4d --output_directory "$EGO4D_ROOT" --datasets annotations --benchmarks em fho --version v2 -y

# precomputed clip features used by the MQ/NLQ heads (the big one)
ego4d --output_directory "$EGO4D_ROOT" --datasets omnivore_video_swinl --version v2 -y

echo "download step done. Features at $EGO4D_ROOT/v2/omnivore_video_swinl/"
du -sh "$EGO4D_ROOT"/v2/* 2>/dev/null || true
