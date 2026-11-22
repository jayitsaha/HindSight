# Full run: SuperMemory-VQA, Object & Location Memory

Outputs of the full PACE run (82 sessions, 924 questions) and the abstention classifier. `SUMMARY.md` has the
headline tables; `jobs.txt` lists the Slurm jobs that produced them.

- `curation/`: evidence coverage of the curated frames.
  - `coverage_summary.csv`, `coverage_per_session.csv`: coverage per method and K, overall and per session.
  - `coverage_vs_K.png`, `span_coverage_vs_K.png`, `coverage_by_session_K100.png`: coverage plots.
  - `timelines/`: one plot per session (69 sessions with evidence) showing kept frames against evidence spans, K = 100.
  - `grids/`: a few example KMeans K = 100 keyframe grids (all 69 are on the Hugging Face dataset
    `hindsight-cs7641/hindsight-curated-frames`).
- `vlm/<run>/`: Gemma-4-E4B answers for `curated_kmeans_K100`, `curated_uniform_K100` and `blind` (no frames):
  `eval_report.md` and `metrics.json` (metrics), `predictions.jsonl` / `raw_predictions.jsonl` (per question,
  including option log-probs), `run_config.json`.
- `abstention/`: the answer-or-abstain classifier (`python -m hindsight.evaluation.abstention`):
  `ABSTENTION.md` and `abstention_results.json` (metrics per run), `<run>_decisions.jsonl` (per question score,
  the fold's tuned cut-off `tau` and the decision), `<run>_risk_coverage.png`.
