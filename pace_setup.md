# Georgia Tech PACE ICE Cluster Guide

This guide details how to sync and run the Hindsight VLM pipeline on Georgia Tech's **PACE ICE** (Instructional Cluster Environment) with your account (`vchheda7`).

---

## 1. Rsync Local Codebase to PACE ICE

From your **local Mac terminal** (inside your project directory `/Users/veerchheda/coding/cs7641proejct`):

### Standard Rsync:
```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' \
  /Users/veerchheda/coding/cs7641proejct/ \
  vchheda7@login-ice.pace.gatech.edu:/home/hice1/vchheda7/scratch/cs7641project/
```

### Or using an active SSH ControlMaster socket (skip Duo password prompt):
```bash
# 1. Establish socket once in terminal:
ssh -M -S ~/.ssh/pace.sock vchheda7@login-ice.pace.gatech.edu

# 2. Rsync anytime without re-authenticating:
rsync -avz -e "ssh -S ~/.ssh/pace.sock" --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' \
  /Users/veerchheda/coding/cs7641proejct/ \
  vchheda7@login-ice.pace.gatech.edu:/home/hice1/vchheda7/scratch/cs7641project/
```

---

## 2. Connect to PACE ICE

```bash
ssh vchheda7@login-ice.pace.gatech.edu
```

Navigate to your scratch directory:
```bash
cd /home/hice1/vchheda7/scratch/cs7641project
```

---

## 3. Set Up Conda Environment on Scratch

PACE `$HOME` has a strict quota (typically 5–10 GB). Always create your environment on your scratch path:

```bash
module load anaconda3

# Create an isolated environment directly inside scratch. NumPy 1.x avoids
# mixing PACE's base SciPy with user-site NumPy 2.x packages.
conda create --prefix ./env-vlm python=3.10 pip "numpy<2" scipy -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/env-vlm"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
```

For an **NVIDIA GPU**, load the CUDA module and install its PyTorch build:

```bash
module load cuda/12.1 2>/dev/null || module load cuda
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the inference, dataset, and video dependencies in the same environment:

```bash
python -m pip install "transformers==4.52.4" "datasets==4.8.5" accelerate opencv-python av pillow pydantic
python -c 'import torch; from transformers import AutoProcessor, AutoModelForVision2Seq; print(torch.__version__, "imports OK")'
```

For the existing SmolVLM environment, keep Transformers on the 4.x API (`4.52.4`) with the CUDA 12.1 PyTorch wheel (`2.5.1`). The backend now supports both the old `AutoModelForVision2Seq` and newer image-text auto class for this model. Gemma 4 needs a separate Transformers 5 environment and `AutoModelForMultimodalLM`.

Keep Datasets at `4.8.5` as well: Datasets 5.2 requires Hugging Face Hub `>=1.31`, which conflicts with the Hub `<1.0` range required by Transformers 4.52.4.

If your allocated GPU is an AMD MI210, do not install the CUDA wheel above; use a PACE-supported ROCm PyTorch build instead. The sample submission script activates `$SCRATCH_PROJECT/env-vlm` and disables user-site packages.

### Optional: Gemma 4 E4B on an NVIDIA H200

Gemma 4 needs the newer Transformers multimodal API, so keep it in its own environment rather than upgrading the working `env-vlm`. E4B is the smaller Gemma 4 vision-and-audio model; the official BF16 inference estimate is about 17.9 GB of GPU memory. The sample script uses 32 GB of host RAM and a one-hour limit for E2B/E4B.

```bash
module load anaconda3
export SCRATCH_PROJECT=/home/hice1/vchheda7/scratch/cs7641project
export HF_HOME="$SCRATCH_PROJECT/hf_cache"
export CONDA_PKGS_DIRS="$SCRATCH_PROJECT/conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH_PROJECT/pip_cache"
mkdir -p "$HF_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

# Confirm this path is charged to ICE Scratch before large downloads.
readlink -f "$SCRATCH_PROJECT"
df -h "$SCRATCH_PROJECT"

conda create --prefix "$PWD/env-gemma4" python=3.11 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/env-gemma4"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip

# CUDA 12.8 PyTorch wheel; H200 needs an NVIDIA GPU allocation.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==5.22.2 accelerate av pillow pydantic datasets==4.8.5
python -c 'import torch; from transformers import AutoProcessor, AutoModelForMultimodalLM; print(torch.__version__, torch.cuda.is_available(), "Gemma 4 imports OK")'
```

If the model page requires access approval, accept its terms while signed into Hugging Face, then authenticate on PACE (`hf auth login`). Keep the model cache in scratch, and check free space before downloading E4B (reserve roughly 20 GB for its BF16 weights and files):

```bash
pace-quota
```

Download the model from a login node before allocating a GPU, then submit the smoke test using the new environment:

```bash
hf download google/gemma-4-E4B-it
HINDSIGHT_ENV=/home/hice1/vchheda7/scratch/cs7641project/env-gemma4 \
  bash scripts/submit_pace_sample.sh google/gemma-4-E4B-it
```

The `E` models have more stored parameters than their effective parameter names imply. Gemma 4 E4B has about 8B total parameters and 4.5B effective parameters; Google estimates 17.9 GB of GPU memory for BF16 inference, before extra context/frame overhead. See the [official model card](https://huggingface.co/google/gemma-4-E4B-it) and [memory table](https://ai.google.dev/gemma/docs/core).

---

## 4. Hugging Face Cache Setup

Always keep model caches on your scratch storage:

```bash
export HF_HOME=/home/hice1/vchheda7/scratch/cs7641project/hf_cache
mkdir -p $HF_HOME
```

---

## 5. Check Available GPUs & Submit (No `salloc`)

To test the pipeline end-to-end on **PACE ICE** with a fast, lightweight non-Chinese model:

```bash
bash scripts/submit_pace_sample.sh
```
The submit script selects a reported NVIDIA GPU GRES (H200, H100, A100, A40, or RTX 6000) and refuses to submit if it finds none. It does not use generic `gpu:1`, which can assign an AMD MI210 incompatible with the CUDA PyTorch build.

By default, it uses **`HuggingFaceTB/SmolVLM-Instruct`** (2.2B, non-Chinese).

To run larger models once the test passes:
```bash
# Pixtral 12B (Mistral AI - France):
bash scripts/submit_pace_sample.sh mistralai/Pixtral-12B-2409
```

This script:
1. Queries `sinfo` for idle H200/H100/A100 nodes.
2. Allocates GPU resources with `--gres`.
3. Submits via `sbatch`.
4. Prints the Job ID and real-time log monitoring command:
   ```bash
   tail -f /home/hice1/vchheda7/scratch/cs7641project/logs/sample_<JOB_ID>.out
   ```
