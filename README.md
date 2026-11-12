# HINDSIGHT

Egocentric object and action memory with calibrated recall, for people with
cognitive impairment. HINDSIGHT watches first-person video, keeps a timeline of
objects and actions, and answers spoken questions — refusing when it isn't sure,
because a user with memory loss tends to believe whatever the assistant says.

It does four things:

- **Object retrieval** — "where is my wallet?" returns the last frame and box where
  the wallet appeared. A trained match scorer beats plain cosine on viewpoint
  changes and look-alike objects.
- **Put-down detection** — notices when an object leaves the hand and warns if it was
  set down somewhere unusual (misplacement), before you go looking for it.
- **Action + repetition** — labels what you're doing and warns when you repeat the
  same action within minutes, a common early sign of dementia.
- **Calibrated assistant** — answers timeline questions and says "I don't know" when
  the evidence is weak, scored by selective-classification risk/coverage.

This repo is a CS 7641 course project. The heavy perception models run on a server;
an Android app (React Native / Expo) is the capture-and-voice client, standing in for
smart glasses we don't have yet.

## How it's put together

```
Android app (Expo)            server (FastAPI)
  camera frames  ───────────▶  ingest ─▶ retrieval ─▶ timeline
  voice question ───────────▶  assistant (calibrated refusal)
  alerts/answers ◀───────────  put-down + action detectors ─▶ alerts
```

The ML contribution is the trained scorers (LogisticRegression, RandomForest, MLP,
GradientBoosting) and clustering (KMeans, HDBSCAN) layered on top of precomputed
embeddings. Foundation models sit behind a provider interface:

| role                    | NVIDIA NIM model                    | local fallback          |
|-------------------------|-------------------------------------|-------------------------|
| image/text embeddings   | `nvidia/nvclip`                     | open_clip ViT-B/32      |
| open-vocab detection    | `microsoft/kosmos-2`                | window proposals        |
| answer-checking VLM     | `meta/llama-3.2-90b-vision-instruct`| retrieval-margin heuristic |
| speech-to-text          | on-device                           | on-device               |

If NIM keys don't authorize (or you're offline), everything degrades to the local
path and still runs end-to-end — the course's ML experiments never depend on NIM.

## Data

Two tracks. The fast one is a **synthetic egocentric generator**
(`hindsight.ingest.synthetic`) with planted structure and ground truth for all four
modules — it runs on a laptop and powers CI and the app demo.

The real one trains on **Ego4D** (`hindsight.benchmarks`): four heads on the
precomputed clip features — ActionFormer-lite for Moment Queries, a VSLNet-lite
localizer for NLQ, a PNR/state head for hands-objects, and a VQ2D match-localizer.
The foundation backbone stays frozen (Ego4D ships its features), so the models you
*train* are the whole ML contribution. Every trainer has a `--smoke` mode that runs
on CPU with no data:

```bash
python -m hindsight.benchmarks.train mq --smoke     # 10 CPU steps on synthetic features
```

Run it for real on GPU (Georgia Tech PACE, with 8h-limit checkpoint/resume) per
[`docs/PACE.md`](docs/PACE.md); dataset loaders and the download in
[`docs/EGO4D.md`](docs/EGO4D.md).

## Frame curation (SuperMemory-VQA)

The current real-data track: choose K keyframes from long egocentric sessions with **uniform sampling** or
**CLIP + KMeans** (`hindsight.video.curation`), and feed them to a VLM. Batch runs on PACE with
`hindsight.curation` + `pace/scripts/curate_array.sbatch`. Guide: [`docs/FRAME_CURATION.md`](docs/FRAME_CURATION.md),
results: [`docs/RESULTS.md`](docs/RESULTS.md#frame-curation-on-supermemory-vqa-real-data-pace), notebooks: `notebooks/`.

## Quick start

```bash
# server
python3.14 -m venv .venv && ./.venv/bin/pip install -e server
./.venv/bin/python -m hindsight.cli gen            # materialize synthetic data
./.venv/bin/python -m hindsight.training.eval_all  # train + evaluate every module
./.venv/bin/uvicorn hindsight.api.app:app --reload # serve the API

# app
cd app && npm install && npx expo start
```

Copy `.env.example` to `.env` and add NVIDIA NIM keys to use the hosted models;
leave it empty to run fully local.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design,
[`docs/RESULTS.md`](docs/RESULTS.md) for the numbers, and
[`docs/API.md`](docs/API.md) for the HTTP/WS reference.
