# Results

Trained on `train_01`, evaluated on `val_01` (synthetic, seeds 7 and 11),
reproducible with `python -m hindsight.training.eval_all`. These are the numbers
the harness prints; nothing here is hand-picked, and the places where our method
only ties a baseline are called out rather than hidden.

## Retrieval — object query → correct crop

| scorer | Recall@1 | MRR | AUC |
|--------|---------|-----|-----|
| cosine | 0.625 | 0.771 | 0.844 |
| logreg | 1.000 | 1.000 | 0.919 |
| random forest | 1.000 | 1.000 | 0.997 |
| mlp | 1.000 | 1.000 | 0.936 |
| KMeans index | 0.625 (brute 0.625) | | |

The headline result: a trained match metric beats cosine (Recall@1 0.625 → 1.0).
Cosine is dominated by shared appearance and confuses look-alike objects; the
learned scorers lean on the clean identity dimensions. The KMeans ANN index costs
zero Recall@1 against brute force. (The identity dims carry light noise on purpose,
so RF lands at 0.997 rather than a suspicious 1.0.)

## Put-down and misplacement

| metric | value |
|--------|-------|
| PNR detection AUROC | 0.994 |
| state-change accuracy | 0.917 |
| misplacement AUROC (clustering) | 0.655 |
| misplacement AUROC (counting baseline) | 0.714 |

PNR (the release frame) and state-change (the window around it) are distinct
targets from the hand-object features, both predicted well. Misplacement is the
honest soft spot: on clean, discrete synthetic rooms, naive per-location counting
is near-optimal, so the scene-clustering model only matches it (within 0.06). The
clustering approach is built to pay off on sparse, continuous real scenes — which
we don't have — so we report the tie instead of engineering a win.

## Action and repetition

| metric | value |
|--------|-------|
| classification macro-F1 | 0.934 |
| per-window accuracy | 0.959 |
| temporal localization mAP | 0.890 |
| repetition AUROC (learned) | 0.741 |
| repetition AUROC (same-class rule) | 0.500 |

Classification isn't a perfect 1.0 because confusable activities (eating/drinking,
typing/writing) genuinely overlap. Repetition is the harder result: the learned
pairwise scorer beats the same-class rule using only instance similarity, never the
time gap — on 27 positive pairs out of 117 same-label candidates, so read it as a
demonstration, not a tight estimate.

## Clustering (unsupervised, vs true labels)

| | NMI | ARI |
|--|-----|-----|
| object identity (crops) | 0.150 | 0.101 |
| action type (windows) | 0.877 | 0.731 |

Action windows cluster cleanly; object crops barely cluster at all. That gap is the
point — viewpoint and appearance noise bury object identity in the raw embeddings,
which is exactly why retrieval needs a supervised metric rather than clustering.

## Assistant — calibrated refusal

| metric | value |
|--------|-------|
| coverage @ 80% accuracy | 0.963 |
| AURC (model / random) | 0.027 / 0.167 |
| refusal recall on unanswerable (n=6) | 1.000 |
| coverage / accuracy on answered | 0.778 / 1.000 |

Correctness here means the located answer is actually right (label/window match),
not merely that the assistant committed. It answers every answerable question
correctly and refuses every unanswerable one. On synthetic data the two are cleanly
separable, so coverage is high; on real Ego4D NLQ, where the best systems find the
right window 20–30% of the time, we'd expect far lower coverage at the same
restraint — the mechanism is the contribution, not the headline coverage.

## Frame curation on SuperMemory-VQA (real data, PACE)

Keyframe selection for the VLM stage: `hindsight.video.curation` (guide: [FRAME_CURATION.md](FRAME_CURATION.md)),
run with `pace/scripts/curate_array.sbatch`, scored with `python -m hindsight.curation.analyze`.

**Setup.** 30 SuperMemory-VQA sessions (3 per participant, those with the most *Object & Location Memory* answer
evidence). Frames sampled at 1 fps, embedded with CLIP ViT-B/32 (open_clip, laion2b), KMeans with K clusters,
the frame nearest each centre kept and sorted by time. Baselines with the same K: uniform (evenly spaced) and
random (mean of 20 seeds). L40S on PACE ICE.

**Metric.** Evidence coverage: share of answer-evidence moments with at least one kept frame inside the
evidence span (±2 s; frames after the question time don't count when the evidence is in the question's own session).
Rephrased questions often share a span, so we report **unique moments** (266) as the main number; all
503 question spans are in `results/curation/pace_30_sessions/coverage_summary.csv`.

| K (frames kept) | Random | Uniform | KMeans |
|---|---|---|---|
| 32  | 36.2 | 45.9 | **46.6** |
| 64  | 57.4 | **79.7** | 70.7 |
| 100 | 71.7 | **90.6** | 80.8 |
| 150 | 81.8 | **97.4** | 91.0 |

KMeans beats random at every K and ties uniform at K = 32, but uniform wins as K grows. KMeans spends frames on
visually distinct scenes; short moments inside long, similar-looking stretches (e.g. putting a phone down in
the same room) merge into one cluster whose representative comes from another time. It uses neither the
question nor recency, which motivates question-aware selection (CLIP top-K / supervised scorer) next.

Caveats: coverage is measured only in the session holding the evidence (oracle session selection), and
unanswerable questions have no evidence session. Figures: `results/curation/pace_30_sessions/`
(`coverage_vs_K.png`, `coverage_by_session_K100.png`, `timelines/`, example `grids/`). The keyframe videos
are on Hugging Face (private, team org).
