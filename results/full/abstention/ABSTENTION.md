# Abstention classifier (leave-one-participant-out)

Answer with Gemma's most likely real option when the classifier says so, otherwise abstain. Every number is on held-out participants; the threshold is tuned on the training participants only.

## curated_kmeans_K100_lp

924 questions, 10 participants. Gemma QA-MRR from the log-probs: 60.4%. Answering with the best real option would be right on 21.9% of questions.

| metric | Gemma (own choice) | Tuned threshold (baseline) | Logistic regression |
|---|---|---|---|
| QA accuracy | 38.6% | 40.3% | 39.3% |
| Score w/ partial credit | 41.8% | 47.2% | 44.8% |
| Answerable accuracy | 10.8% | 18.1% | 13.5% |
| Abstention accuracy | 92.4% | 83.2% | 89.2% |
| Answerability F1 (answerable) | 38.9% | 64.1% | 54.0% |
| Answerability macro-F1 | 46.9% | 62.0% | 56.1% |
| Hallucination rate | 7.6% | 16.8% | 10.8% |
| Over-abstention rate | 74.9% | 48.8% | 60.9% |
| Answered (coverage) | 19.2% | 39.5% | 29.4% |
| Coverage at 80% selective accuracy | 0.0% | 0.0% | 0.0% |
| AURC (lower is better) | 0.733 | 0.695 | 0.711 |
| McNemar p vs Gemma (accuracy) | - | 0.101 | 0.418 |

Logistic regression, standardised coefficients (positive -> answer): `log_odds` +0.60, `p_abstain` +0.35, `gemma_abstained` -0.16, `entropy` -0.16, `abstain_pos_2` -0.15, `letter_mass` -0.15, `p_top1` +0.14, `p_top1_renorm` -0.14

Risk-coverage plot: `curated_kmeans_K100_lp_risk_coverage.png`; per-question decisions: `curated_kmeans_K100_lp_decisions.jsonl`.

## curated_uniform_K100_lp

924 questions, 10 participants. Gemma QA-MRR from the log-probs: 60.1%. Answering with the best real option would be right on 21.3% of questions.

| metric | Gemma (own choice) | Tuned threshold (baseline) | Logistic regression |
|---|---|---|---|
| QA accuracy | 38.0% | 39.3% | 39.6% |
| Score w/ partial credit | 41.2% | 45.9% | 45.5% |
| Answerable accuracy | 9.5% | 15.9% | 16.3% |
| Abstention accuracy | 93.0% | 84.4% | 84.8% |
| Answerability F1 (answerable) | 37.6% | 60.6% | 59.9% |
| Answerability macro-F1 | 46.2% | 59.7% | 59.3% |
| Hallucination rate | 7.0% | 15.6% | 15.2% |
| Over-abstention rate | 76.0% | 53.0% | 53.9% |
| Answered (coverage) | 18.2% | 36.3% | 35.6% |
| Coverage at 80% selective accuracy | 0.0% | 0.6% | 0.1% |
| AURC (lower is better) | 0.745 | 0.703 | 0.708 |
| McNemar p vs Gemma (accuracy) | - | 0.175 | 0.167 |

Logistic regression, standardised coefficients (positive -> answer): `log_odds` +0.70, `p_top1_renorm` -0.26, `p_abstain` +0.24, `entropy` -0.21, `letter_mass` -0.17, `abstain_pos_0` +0.13, `margin` +0.11, `abstain_pos_2` -0.11

Risk-coverage plot: `curated_uniform_K100_lp_risk_coverage.png`; per-question decisions: `curated_uniform_K100_lp_decisions.jsonl`.

## blind_lp

924 questions, 10 participants. Gemma QA-MRR from the log-probs: 57.7%. Answering with the best real option would be right on 21.3% of questions.

| metric | Gemma (own choice) | Tuned threshold (baseline) | Logistic regression |
|---|---|---|---|
| QA accuracy | 34.1% | 33.0% | 36.5% |
| Score w/ partial credit | 34.1% | 33.3% | 39.1% |
| Answerable accuracy | 0.0% | 1.5% | 13.1% |
| Abstention accuracy | 100.0% | 94.0% | 81.6% |
| Answerability F1 (answerable) | 0.0% | 8.2% | 48.2% |
| Answerability macro-F1 | 25.4% | 28.9% | 50.6% |
| Hallucination rate | 0.0% | 6.0% | 18.4% |
| Over-abstention rate | 100.0% | 95.6% | 65.2% |
| Answered (coverage) | 0.0% | 5.0% | 29.2% |
| Coverage at 80% selective accuracy | 0.1% | 0.0% | 0.0% |
| AURC (lower is better) | 0.783 | 0.781 | 0.717 |
| McNemar p vs Gemma (accuracy) | - | 0.0872 | 0.0735 |

Logistic regression, standardised coefficients (positive -> answer): `letter_mass` +0.82, `log_odds` +0.56, `entropy` -0.35, `abstain_pos_2` -0.34, `abstain_pos_0` +0.26, `log_q_time` -0.22, `abstain_pos_1` +0.15, `margin` -0.12

Risk-coverage plot: `blind_lp_risk_coverage.png`; per-question decisions: `blind_lp_decisions.jsonl`.

