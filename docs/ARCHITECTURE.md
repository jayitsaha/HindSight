# Architecture

## Shape

Two processes. A FastAPI **server** holds all perception and memory; a React Native
**app** captures first-person video and voice and renders the timeline and alerts.
Between them, the HTTP/WS contract in [API.md](API.md).

Inside the server, a frame flows:

```
frame ─▶ grounder (candidate boxes) ─▶ embedder (NV-CLIP) ─▶ detections
                                                     │
          ┌──────────────────────────────┬──────────┴───────────┐
          ▼                              ▼                        ▼
   retrieval index            put-down detector           action detector
   (KMeans + scorer)      (GradientBoosting PNR +        (MLP labels +
          │                HDBSCAN misplacement)          pairwise repeat)
          ▼                              ▼                        ▼
                          timeline store (sqlite)  ─────▶  alerts
                                   │
   question ─▶ assistant ─▶ retrieve window ─▶ VLM check ─▶ refuse or answer
```

## Why a provider layer

The proposal names specific foundation models (SigLIP 2, Grounding DINO,
InternVideo2, Qwen2.5-VL, Whisper). We don't host those; we reach equivalent
models through NVIDIA NIM, and we must keep running when NIM is unreachable. So
every heavy call goes through `providers/`:

- `base.py` — `Embedder`, `Grounder`, `VLM`, `ASR` protocols.
- `nim.py` — NIM client with a key pool that round-robins and benches a key on
  401/403/429, so one bad or throttled key doesn't stall the system.
- `local.py` — open_clip embedder, window-proposal grounder, a heuristic VLM that
  answers from retrieval evidence, on-device ASR.
- `registry.py` — picks NIM when a key authorizes (probed once, cached), else local.

Modules never import a concrete provider; they call `get_embedder()` and friends.
This is also what makes the whole thing run on a laptop with no GPU.

## Why the ML lives in scorers, not the big models

The foundation models give embeddings and features; they are frozen. The learning
the course cares about is the thin trained layer on top:

- **retrieval** — a learned match metric over `[q·c, |q−c|]` pair features
  (LogReg/RF/MLP) that beats cosine on viewpoint and look-alikes; KMeans for a cheap
  index.
- **put-down** — GradientBoosting over hand-object window features for the
  point-of-no-return; HDBSCAN + a smoothed rest-location histogram for how unusual a
  resting place is.
- **action** — MLP over clip-window features for labels; a pairwise MLP (deliberately
  blind to the time gap) for repetition, beating a same-class rule.
- **assistant** — selective classification: calibrate a confidence threshold on the
  retrieval margin and VLM agreement, report risk/coverage and AURC, refuse below it.

All of it trains and evaluates on the synthetic generator with no real video, and
swaps to Ego4D through `ingest/ego4d.py` without touching the modules.

## Data path

`ingest/synthetic.py` simulates a person moving through rooms, handling objects, and
acting — emitting per-module ground truth and an embedding space engineered so a
learned metric can beat cosine. `ingest/ego4d.py` parses the real Ego4D VQ2D / MQ /
NLQ / PNR annotations and precomputed features into the same shapes.

The live **timeline store** (`timeline/store.py`, sqlite) is the running memory a
session accumulates: frames, detections (with embeddings), put-down events, action
segments, alerts, answers.

## Calibrated refusal

The point of the assistant is restraint. For each question it proposes an answer from
retrieval, scores how strong the top match is, optionally asks the VLM whether the
frames support it, and combines these into a confidence. A threshold tuned on held-out
questions (half of NLQ-val in the real setting) decides answer-vs-refuse; we report
coverage at a target accuracy and AURC so the trade-off is explicit rather than vibes.
