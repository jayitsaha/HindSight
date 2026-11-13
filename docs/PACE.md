# Running HINDSIGHT on PACE (real Ego4D)

Georgia Tech **PACE ICE**, SLURM scheduler, H100/H200 with ~8h job limits. This is
the real-data path: download Ego4D's precomputed features + annotations to scratch,
train the four benchmark heads on GPU, evaluate against the official splits. The
foundation models are frozen (Ego4D ships their features) — the ML is the heads you
train here.

> Confirm with CS 7641 staff / `pace-check-queue -s`: `GTUSER`, `-A ACCOUNT`
> (class allocation, often `gts-<pi>`), `-q QOS`, and the exact `--gres=gpu:<type>:1`.
> Wrong account/partition is the #1 reason a job won't start.

## 1. Access and layout

```bash
ssh GTUSER@login-ice.pace.gatech.edu            # GT VPN off-campus
mkdir -p ~/scratch/hindsight/{ego4d,envs,runs,logs,ckpt}
```

Home is small and backed up (code, env). **Scratch** is large and NOT backed up —
Ego4D features and checkpoints go there. Copy final artifacts off before the purge.

## 2. Environment

```bash
module load anaconda3 cuda
conda create -y -p ~/scratch/hindsight/envs/hs python=3.10
conda activate ~/scratch/hindsight/envs/hs
python -m pip install --upgrade pip
# CUDA PyTorch wheel (match the cluster CUDA from `nvidia-smi`; cu121 is a safe default):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
# the repo (code only; rsync it up first — see §6) + the Ego4D CLI:
python -m pip install -e ~/scratch/hindsight/HINDSIGHT/server
python -m pip install ego4d
```

Reactivate each session: `module load anaconda3 cuda && conda activate ~/scratch/hindsight/envs/hs`.

## 3. Get Ego4D (license first, then download)

1. Sign the license at <https://ego4d-data.org/docs/start-here/> and get your **AWS
   access key / secret** from the approval email. Configure them:
   ```bash
   aws configure            # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
   ```
2. Download only what the four benchmarks need, to scratch. The precomputed
   **omnivore** video features are what ActionFormer/VSLNet train on — this is the
   big one (plan for a few hundred GB; subset with `--video_uids` to start smaller):

```bash
export EGO4D_ROOT=~/scratch/hindsight/ego4d
# annotations for all four benchmarks (~3 GB)
ego4d --output_directory "$EGO4D_ROOT" --datasets annotations \
      --benchmarks em fho --version v2 -y
# precomputed clip features (omnivore_video_swinl): MQ + NLQ localization heads
ego4d --output_directory "$EGO4D_ROOT" --datasets omnivore_video_swinl --version v2 -y
# VQ2D needs the query-object frames; start with a subset of clips:
ego4d --output_directory "$EGO4D_ROOT" --datasets clips --benchmarks vq2d --version v2 \
      --video_uids $(head -40 docs/ego4d_vq2d_uids.txt) -y
```

Run the download in an sbatch job (it's long) — see `pace/scripts/download.sbatch`.
Resulting layout the loaders expect:

```
$EGO4D_ROOT/v2/
  annotations/{moments_train.json, nlq_train.json, vq_train.json, fho_oscc-pnr_train.json, ...}
  omnivore_video_swinl/<clip_uid>.pt      # (T, 1536) per clip
  clips/<video_uid>.mp4                    # only for VQ2D crops
```

## 4. Smoke-test, then train

Everything runs on tiny random tensors with `--smoke` first (no data, CPU) so you
catch bugs before burning GPU hours:

```bash
python -m hindsight.benchmarks.train mq --smoke        # 10 steps on fake features
```

Then submit the real jobs (each checkpoints every N steps and resumes, so an 8h
wall limit is fine — resubmit with `--resume`):

```bash
sbatch pace/scripts/train_mq.sbatch        # action localization (MQ)
sbatch pace/scripts/train_nlq.sbatch       # moment retrieval (NLQ)
sbatch pace/scripts/train_pnr.sbatch       # put-down / state change (OSCC-PNR)
sbatch pace/scripts/train_vq2d.sbatch      # visual query (VQ2D)
```

Monitor: `squeue --me -l` · `tail -f ~/scratch/hindsight/logs/mq-*.out` ·
`srun --jobid JOBID --overlap --pty nvidia-smi`.

## 5. Evaluate and collect

```bash
python -m hindsight.benchmarks.eval mq --checkpoint ~/scratch/hindsight/ckpt/mq/best.pt
# writes runs/mq/metrics.json (official-style mAP etc.)
```

Pull results back to the laptop (§6) for the report.

## 6. Move code up / results down

```bash
# from the LAPTOP. rsync only creates the final dir, not missing parents — make it first:
ssh GTUSER@login-ice.pace.gatech.edu "mkdir -p ~/scratch/hindsight/HINDSIGHT"
rsync -avP \
  --exclude '.venv' --exclude '.git' --exclude 'node_modules' \
  --exclude 'server/data' --exclude 'server/artifacts' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.expo' \
  ~/Documents/COURSES/SEM1/ML/PROJECT/HINDSIGHT/ \
  GTUSER@login-ice.pace.gatech.edu:~/scratch/hindsight/HINDSIGHT/
scp GTUSER@login-ice.pace.gatech.edu:~/scratch/hindsight/runs/*/metrics.json ./results/
```

The app and `node_modules` stay on the laptop — PACE only runs the Python server.

Never upload `.env` or AWS keys; keep secrets in `chmod 600` files read at runtime.

## 7. Frame curation on SuperMemory-VQA

Writes KMeans and same-size uniform keyframe videos per session for the VLM stage (guide:
[`FRAME_CURATION.md`](FRAME_CURATION.md)). No Ego4D license needed; videos download from Hugging Face.

```bash
# once: env + Hugging Face login (videos and CLIP weights cache on scratch)
module load anaconda3 && conda activate ~/scratch/hindsight/env
pip install -e "$HINDSIGHT_REPO/server[curation]"      # or: export PYTHONPATH=$HINDSIGHT_REPO/server
export HF_HOME=~/scratch/hindsight/hf && hf auth login

# pick sessions, submit one array task per session (run from the work dir)
cd ~/scratch/hindsight
python -m hindsight.curation.make_sessions --n 30 --per-person 3 --out sessions.txt
mkdir -p logs && sbatch --array=0-$(( $(wc -l < sessions.txt) - 1 ))%10 "$HINDSIGHT_REPO/pace/scripts/curate_array.sbatch"
squeue --me

# when done (login node, no GPU): tables + graphs
python -m hindsight.curation.analyze --out out --results results --k-bar 100
```

`HINDSIGHT_REPO` defaults to `~/scratch/hindsight/HindSight` in the sbatch script. Outputs go to
`out/<session>/{kmeans,uniform}_K<k>.{mp4,json}`; embeddings cache in `cache/`. Request an NVIDIA GPU type
explicitly (`--gres=gpu:l40s:1`, `v100`, `rtx_6000`): `mi210` and `r9700` are AMD. A 15.7-min session takes
about 135 s for KMeans (embed at 1 fps) on an L40S; reading the 1408×1408 video dominates.

## 8. VLM evaluation (Gemma 4) on SuperMemory-VQA

Answers Object & Location Memory questions from curated frames (guide: [`VLM_PIPELINE.md`](VLM_PIPELINE.md)).
Gemma 4 needs Transformers 5, so it gets its own env, separate from the curation env.

```bash
# once: Gemma 4 env on scratch (CUDA 12.8 wheel; needs an NVIDIA GPU at run time)
module load anaconda3
export HF_HOME=~/scratch/hindsight/hf CONDA_PKGS_DIRS=~/scratch/hindsight/conda_pkgs PIP_CACHE_DIR=~/scratch/hindsight/pip_cache
conda create --prefix ~/scratch/hindsight/env-gemma4 python=3.11 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ~/scratch/hindsight/env-gemma4
export PYTHONNOUSERSITE=1
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5" accelerate av pillow pydantic opencv-python-headless huggingface_hub
python -c 'from transformers import AutoProcessor, AutoModelForMultimodalLM; print("Gemma 4 imports OK")'

# accept the model terms on huggingface.co while signed in, then download from the login node (~20 GB)
hf auth login && hf download google/gemma-4-E4B-it

# smoke test (synthetic video -> answer)
HINDSIGHT_ENV=~/scratch/hindsight/env-gemma4 bash "$HINDSIGHT_REPO/pace/scripts/submit_vlm_smoke.sh" google/gemma-4-E4B-it

# curated frames vs blind control (from the work dir); curated videos come from the HF dataset or out/
cd ~/scratch/hindsight && mkdir -p logs
CUR=hf:hindsight-cs7641/hindsight-curated-frames
for M in kmeans uniform; do
  sbatch "$HINDSIGHT_REPO/pace/scripts/vlm_eval.sbatch" --condition curated --curated-dir $CUR --method $M --k 100 --image-tokens 140
done
sbatch "$HINDSIGHT_REPO/pace/scripts/vlm_eval.sbatch" --condition blind --curated-dir $CUR --method kmeans --k 100
python -m hindsight.evaluation.compare results/vlm/curated_kmeans_K100 results/vlm/curated_uniform_K100 results/vlm/blind

# Veer's oracle pilot (40 Person_1 questions, frames cut around the evidence)
HINDSIGHT_ENV=~/scratch/hindsight/env-gemma4 bash "$HINDSIGHT_REPO/pace/scripts/submit_pilot.sh"
```

On the 30 curated sessions, 600 questions qualify (455 answerable, 145 unanswerable), with up to 200 frames per
question at K=100 (questions whose evidence spans two sessions). Gemma 4 E4B in bf16 is ~18 GB plus frames, so an
L40S (48 GB) is enough. The first question logs `N frames -> T tokens`: check that T moves with `--image-tokens`.
Keep the older SmolVLM env (`transformers==4.52.4`, `datasets==4.8.5`, CUDA 12.1 torch 2.5.1) separate if you still
need it; Transformers 4 and 5 do not mix.

## 9. Whole dataset in one go

`pace/scripts/run_full_pipeline.sh` runs on the login node, from the work dir that holds `out/`. It checks the
envs, the model cache and the questions file, then queues a chain of jobs:

1. **Curation:** an array job over every session an Object & Location question needs that isn't already in `out/`.
2. **Gemma 4:** curated KMeans, curated uniform and blind runs on every question whose sessions are curated. Each
   run gets a continuation job that resumes if the first one hits the 8 h limit.
3. **Finalize:** coverage over all sessions plus the VLM comparison, written to `results/full/SUMMARY.md`.

```bash
cd ~/scratch/hindsight
DRY_RUN=1 bash "$HINDSIGHT_REPO/pace/scripts/run_full_pipeline.sh"   # pre-flight + print the sbatch commands
bash "$HINDSIGHT_REPO/pace/scripts/run_full_pipeline.sh"             # queue everything
cat results/full/SUMMARY.md                                          # when it is done
```

Each step starts `afterany` the previous one, so one failed session doesn't block the rest. Running the script again
skips finished sessions, finished K values and answered questions. Settings (`KS`, `VLM_KS`, `IMAGE_TOKENS`,
`CURATION_ENV`, `HINDSIGHT_ENV`, `MODEL`) are listed at the top of the script. One answer-evidence session isn't in
the dataset repo, so the few questions that need it are never scored; the script prints its name.
