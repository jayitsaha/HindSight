# VLM Answering and Evaluation (`hindsight.vlm`, `hindsight.evaluation`)

A vision-language model answers SuperMemory-VQA **Object & Location Memory** multiple-choice questions from a
set of frames, and is scored on accuracy and on whether it abstains correctly. The VLM pipeline, prompt, answer
parser, metrics and pilot were written by Veer Chheda. The curated-frames condition connects it to
[frame curation](FRAME_CURATION.md).

```
frame curation (uniform / CLIP+KMeans)        SuperMemory-VQA question + options
  out/<session>/<method>_K<k>.mp4 + .json                     │
              │  hindsight.evaluation.curated                  │
              ▼  (sessions, cutoff, labels)                    ▼
        [(label, frame), ...]  ───────────────▶  VLMPipeline (Gemma 4 E4B / SmolVLM / mock)
                                                              │  JSON: selected_choice, reasoning, is_unanswerable
                                                              ▼
                                                   Evaluator → metrics.json, eval_report.md
```

## The data

- 4,853 questions in one **test** split (no train/val split on Hugging Face), 10 participants, 82 sessions.
- Object & Location Memory: **933 questions, 613 answerable, 320 unanswerable**. 885 have their evidence in one
  session; 48 need 2–4 sessions.
- Each question is asked at a point in its own (primary) session, given in `question_evidence`. Answers must come
  from what was recorded before then, which can include earlier sessions.
- The top-level `start_time` of a record is the session's Unix start time, not seconds into the video.

## Conditions

`python -m hindsight.evaluation.run --condition <c>` (all options: `--help`):

| condition | what the model sees | needs |
|---|---|---|
| `curated` | the curated K frames of each evidence session, recorded before the question, labelled `[session N, m:ss]` | `--curated-dir out` or `hf:<repo>`, `--method kmeans\|uniform`, `--k 100` |
| `blind` | no frames: shows how far option priors alone get | the same `--curated-dir` (or `--pilot-dir`) as the run it controls for |
| `oracle` | 8 frames cut around the answer-evidence span (upper bound) | `--pilot-dir` from `prepare_pilot` |
| `video` | `<video-dir>/<primary_video_id>.mp4` sampled to `--max-frames` (default 16) | `--video-dir` |

Curated details:
- `--sessions evidence` (default) shows the sessions that hold the answer evidence, or the question's own session
  for unanswerable questions. That is the same oracle session choice the coverage numbers use. `--sessions primary`
  shows only the question's own session.
- Only questions whose sessions were all curated are scored. The 30 PACE sessions cover 600 questions.
- Frames are never subsampled unless you pass `--max-frames`. `--image-tokens 140` sets Gemma 4's per-frame
  visual token budget (70/140/280/560/1120; default 280).
- Frames load one question at a time.

```bash
# local check with the mock backend (no GPU)
python -m hindsight.evaluation.run --backend mock --condition curated --curated-dir out --method kmeans --k 100 --max-questions 5

# on PACE: pace/scripts/vlm_eval.sbatch wraps the same command with --backend hf (docs/PACE.md section 8)
python -m hindsight.evaluation.compare results/vlm/curated_kmeans_K100 results/vlm/curated_uniform_K100 results/vlm/blind
```

Each run writes `metrics.json`, `eval_report.md`, `predictions.jsonl` (per-question correctness),
`raw_predictions.jsonl` (model outputs; a re-run skips these) and `run_config.json`.

## Metrics (`hindsight.evaluation.metrics`)

| metric | meaning |
|---|---|
| accuracy | chose the correct option (on unanswerable questions the correct option is "can not be answered") |
| score with partial credit | 0.5 for the vague option |
| answerable accuracy / abstention accuracy | accuracy on answerable / unanswerable questions |
| hallucination rate | answered when the question was unanswerable |
| over-abstention rate | abstained when it was answerable |
| answerability F1 (answerable, macro) | answer-vs-abstain as a binary decision |
| vague / invalid rate | picked the vague option / output could not be parsed (scored wrong, never guessed) |

`confidence` is `None`. The `hf` backend instead records `option_logprobs`: the log-probability of each option letter
at the step where the model wrote `selected_choice` (no extra forward pass), plus `n_frames`. They feed the
abstention classifier below.

## Abstention classifier (`hindsight.evaluation.abstention`, supervised)

Gemma over-abstains: it picks "can not be answered" on ~75% of answerable questions, yet is right ~70% of the time
when it commits. The classifier learns when to answer instead of trusting that choice.

- **Features** (from `option_logprobs` only): P(abstain option), best and second real option, margin, entropy,
  log-odds best-vs-abstain, total letter mass, position of the abstain option, Gemma's own decision, question time.
  No evidence or session-count features, since those would leak answerability through the oracle session choice.
- **Label:** 1 if answering with Gemma's most likely real option is correct.
- **Policy:** answer with that option if P(label = 1) ≥ τ, otherwise abstain.
- **Model:** LogisticRegression on standardised features. It suits ~900 questions, gives probabilities a threshold
  can act on, and its coefficients show which signals matter. Baseline: a tuned threshold on the log-odds alone
  (one parameter, no learning), so any gain is due to the learned combination of features.
- **Protocol:** leave-one-participant-out over the 10 participants. τ is tuned on the training participants only
  (inner leave-one-participant-out), so every question gets one decision from a model that never saw its participant.
- **Output:** `ABSTENTION.md` (the metrics above plus coverage, AURC, coverage at 80% selective accuracy, McNemar p
  against Gemma's own choice, Gemma's QA-MRR), `abstention_results.json`, per-question decisions, risk-coverage plots.
- **Control:** the blind run. If the classifier gains as much without frames, it is only learning option priors.

```bash
python -m hindsight.evaluation.abstention results/full/vlm/curated_kmeans_K100_lp results/full/vlm/curated_uniform_K100_lp \
    results/full/vlm/blind_lp --out results/full/abstention
```

On PACE, `pace/scripts/submit_abstention.sh` reruns Gemma with log-probs on all three conditions and then the
classifier (4 jobs, about 2 h).

## Python API

```python
from hindsight.vlm.config import VLMConfig
from hindsight.vlm.pipeline import VLMPipeline
from hindsight.evaluation.curated import CuratedFrames

pipe = VLMPipeline(VLMConfig(backend="hf", model_name="google/gemma-4-E4B-it", device="auto",
                             max_frames=None, image_token_budget=140))
cf = CuratedFrames("out", method="kmeans", k=100)
out = pipe.answer_question(question, cf.frames_for(question))   # question: QuestionRecord from SuperMemoryDataLoader
out.predicted_choice, out.abstain, out.reasoning
```

`answer_question` takes:
- a video path (sampled to `max_frames`)
- a list of image paths, PIL images, or `(label, image)` pairs
- one image
- `[]` for text-only

Backends: `hf` (any Hugging Face VLM; Gemma 4 uses `AutoModelForMultimodalLM`) and `mock` (deterministic, for tests).

## Oracle pilot

```bash
python -m hindsight.evaluation.prepare_pilot --subject S1 --n 40 --dry-run   # questions + download plan
python -m hindsight.evaluation.prepare_pilot --subject S1 --n 40             # download, cut frames, delete videos
python -m hindsight.evaluation.run --backend hf --condition oracle --pilot-dir pilot
python -m hindsight.evaluation.run --backend hf --condition blind  --pilot-dir pilot
```

The pilot picks questions that need at most 6 session videos and keeps the answerable/unanswerable mix.
`pace/scripts/submit_pilot.sh` runs all of it as one job.

## Changes from the standalone version

- The package moved under `server/hindsight/`. The config dataclasses live in `hindsight.vlm.config`, because
  `hindsight.config` already holds the server settings. Scripts became `python -m` modules.
- Evidence times parse `MM:SS` strings and both `time_span` / `time_spans` forms (they used to crash).
- `cutoff_timestamp` is the question time from `question_evidence`. It used to be the session's Unix start time.
- The frame cap (previously a hard-coded 16) is configurable and off for curated input. Video files are streamed,
  not decoded whole into RAM.
- A failed Hugging Face load raises an error instead of silently scoring the 5 demo questions.
- The fixed `confidence=0.85` is now `None`, and the Gemini backend was removed (its module did not exist).
