# HindSight full run

## Frame curation
- sessions fully curated (K = 32 64 100 150): **82**
- this run's sessions: 52/52 finished

**unique evidence moments covered (%)**

| K | Random | Uniform | KMeans |
|---|---|---|---|
| 32 | 36.7 | 47.3 | 44.4 |
| 64 | 57.6 | 78.4 | 68.6 |
| 100 | 72.2 | 90.9 | 79.7 |
| 150 | 82.2 | 96.6 | 92.2 |

**evidence spans covered (%)**

| K | Random | Uniform | KMeans |
|---|---|---|---|
| 32 | 35.7 | 42.3 | 44.3 |
| 64 | 56.2 | 73.8 | 69.2 |
| 100 | 70.3 | 85.1 | 79.8 |
| 150 | 80.8 | 96.7 | 92.3 |

Plots and per-session tables: `results/full/curation/` (coverage_vs_K.png, timelines/, grids/)

## Gemma 4 answers


| metric | curated_kmeans_K100 | curated_uniform_K100 | blind |
|---|---|---|---|
| n | 924 | 924 | 924 |
| QA accuracy | 38.6% | 38.0% | 34.1% |
| Score w/ partial credit | 42.1% | 41.3% | 34.1% |
| Answerable accuracy | 10.8% | 9.5% | 0.0% |
| Abstention accuracy | 92.4% | 93.0% | 100.0% |
| Answerability F1 (answerable) | 38.9% | 37.6% | 0.0% |
| Answerability macro-F1 | 46.9% | 46.2% | 25.4% |
| Hallucination rate | 7.6% | 7.0% | 0.0% |
| Over-abstention rate | 74.9% | 76.0% | 100.0% |
| Vague-option rate | 10.5% | 10.2% | 0.0% |
| Invalid rate | 0.0% | 0.0% | 0.0% |

Per-question predictions: `<dir>/predictions.jsonl`; model outputs: `<dir>/raw_predictions.jsonl`.

## Post-midterm: which 100 frames per session? (23 Nov)

Same budget (100 frames per evidence session) and the same Gemma settings as above; only the frame selector changes.
Coverage from `selectors/SELECTORS.md` (677 evidence spans, frames recorded before the question); Gemma runs in
`vlm/curated_agglomerative_K100` and `vlm/selected_logreg_top`.

| | Uniform | KMeans | Temporal agglomerative | Learned picker (logistic regression) | Blind |
|---|---|---|---|---|---|
| type | baseline | unsupervised | unsupervised | supervised, per question | control |
| evidence spans covered | 84.6% | 79.3% | **89.5%** | **89.8%** | – |
| n | 924 | 924 | 924 | 921 | 924 |
| QA accuracy | 38.0% | 38.6% | **39.8%** | 38.9% | 34.1% |
| Score w/ partial credit | 41.3% | 42.1% | **42.8%** | 42.3% | 34.1% |
| Answerable accuracy | 9.5% | 10.8% | **12.5%** | 11.7% | 0.0% |
| Abstention accuracy | 93.0% | 92.4% | 92.7% | 92.0% | 100.0% |
| Answerability F1 (answerable) | 37.6% | 38.9% | 41.5% | **42.3%** | 0.0% |
| Answerability macro-F1 | 46.2% | 46.9% | 48.6% | **48.8%** | 25.4% |
| Hallucination rate | 7.0% | 7.6% | 7.0% | 8.0% | 0.0% |
| Over-abstention rate | 76.0% | 74.9% | 72.9% | **72.1%** | 100.0% |
| Vague-option rate | 10.2% | 10.5% | 9.0% | 10.3% | 0.0% |
| Questions answered (not abstained) | 168 | 177 | 187 | 195 | 0 |
| Right when it answers | 34.5% | 37.3% | **40.6%** | 36.4% | – |

Temporal agglomerative clustering (Ward, only time-adjacent frames merge) covers ~10 points more evidence than KMeans
and gets ~16% more answerable questions right with no rise in hallucination. The learned picker gives the best
answer/abstain balance but slightly more hallucination. Coverage gains of ~10 points turn into only +1-2 points of
accuracy: Gemma often does not use the evidence even when it is shown. The four runs share 920 questions; the
"answered" and "right when it answers" rows are identical on that common set.

Question-grounded retrieval (picking a few windows per question instead of 100 frames) is in `grounded/RECALL.md`
(zero-shot CLIP) and `retriever/RETRIEVER.md` (learned retriever, leave-one-participant-out).
