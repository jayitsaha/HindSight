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
