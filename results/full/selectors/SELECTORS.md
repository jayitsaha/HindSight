# Which 100 frames per session? Evidence coverage by selector

613 answerable questions, 677 evidence spans, 82 sessions. Every selector keeps 100 frames per evidence session. A span is covered if a kept frame is inside it (+-2 s) and recorded before the question. Question-aware selectors keep frames at least 5 s apart and only before the question; their scores are leave-one-participant-out.

| selector | spans covered (%) | questions with a covered span (%) | usable frames per question (median) |
|---|---|---|---|
| Uniform (curated) | 84.6 | 84.7 | 100 |
| KMeans (curated) | 79.3 | 81.1 | 100 |
| Random | 71.2 | 73.0 | 100 |
| Agglomerative (Ward) | 82.4 | 83.0 | 100 |
| Agglomerative, temporal (contiguous) | 89.5 | 90.2 | 100 |
| Zero-shot CLIP top frames | 71.2 | 72.6 | 100 |
| Learned top frames (logistic regression) | 89.8 | 90.7 | 100 |
| Learned top frames (gradient-boosted trees) | 79.6 | 82.1 | 100 |
| 50% KMeans + 50% CLIP top | 85.5 | 87.3 | 100 |
| 50% KMeans + 50% learned (trees) | 83.2 | 85.2 | 100 |

Usable frames = kept frames recorded before the question (frames after it are never shown to Gemma).
