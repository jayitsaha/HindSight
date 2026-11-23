# Learned retriever vs zero-shot CLIP

609 answerable questions (921 scored), sessions = evidence, median 1930 candidate frames per question, 1.54% of candidate frames are evidence. Leave-one-participant-out (10 folds).

**Recall@X**: % of answerable questions with an evidence span (+-2 s) inside one of the first X windows (anchors at least 60 s apart).

## window = +-15 s

| method | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| Zero-shot CLIP (phase 1) | 15.6 | 23.6 | 35.8 | 53.4 |
| Learned: logistic regression | 6.7 | 15.6 | 25.8 | 43.7 |
| Learned: gradient-boosted trees | 11.2 | 19.5 | 32.8 | 45.8 |
| Random anchors | 3.1 | 6.2 | 13.2 | 24.1 |

## window = +-30 s

| method | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| Zero-shot CLIP (phase 1) | 18.7 | 28.1 | 40.7 | 59.1 |
| Learned: logistic regression | 12.3 | 23.0 | 35.1 | 57.3 |
| Learned: gradient-boosted trees | 15.4 | 26.4 | 41.5 | 58.9 |
| Random anchors | 5.0 | 9.6 | 19.6 | 34.6 |

## Frame ranking (is the evidence frame scored high?)

| method | mean AP | AUC | median rank of best evidence frame (percentile, 0 = top) | evidence in top 5% |
|---|---|---|---|---|
| Zero-shot CLIP (phase 1) | 9.76 | 0.611 | 6.7 | 43.5% |
| Learned: logistic regression | 6.52 | 0.643 | 2.1 | 69.0% |
| Learned: gradient-boosted trees | 7.94 | 0.659 | 3.7 | 55.3% |

Logistic regression, largest standardised coefficients (mean over folds): cos_s15 +0.68, cos_s5 -0.56, distinct +0.39, motion +0.36, cos_prompt -0.35, cos_question +0.31, cos_s5_z -0.17, back_0 +0.09, history_pos +0.09, log_min_before +0.08.
