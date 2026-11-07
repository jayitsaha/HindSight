# Synthetic data

`hindsight.ingest.synthetic.generate(video_id, minutes, seed)` simulates a person
moving through rooms, picking up and setting down objects, and doing things, then
emits ground truth for every module. `cli gen` materializes `train_01` (seed 7),
`val_01` (seed 11), and a rendered `demo_01` (seed 23) under `data/synthetic/`.

A `SyntheticVideo` carries:

- **objects** — `dets[Det(frame_idx, t, label, object_id, xyxy, emb_idx)]` with crop
  embeddings `det_emb (N, 64)`, and object queries `queries[Query(...)]` with text
  embeddings `query_emb`. Each query is answerable or not (the refusal label) and
  carries the frame/box/window of the correct answer when it is.
- **put-downs** — `putdowns[PutDown(t, frame_idx, object_label, location_id,
  home_location_id, unusual)]` with a per-event scene embedding `pd_scene_emb (P, 64)`.
  `unusual` is the misplacement label; `ho_feat (M, 7)` are per-second hand-object
  features with `ho_pnr` (the release frame) and `ho_state` (the state-change window).
- **actions** — `segments[ActionSeg(t_start, t_end, label, is_repeat, repeat_of, ...)]`
  with per-segment embeddings `seg_emb (S, 64)` and classifier windows
  `win_feat (W, 96)` / `win_label` / `win_t`.

The embedding space is engineered so a learned match metric beats cosine: each
object's vector splits into high-energy appearance dims (shared across look-alike
pairs, hit with viewpoint noise) and low-energy identity dims (clean,
discriminative). The time gap between actions is deliberately excluded from the
repetition features so the model can't cheat on timing; a repeat instead inherits
the instance embedding of the action it repeats. See the module docstring in
`ingest/synthetic.py` for the exact construction.

Swap to real **Ego4D** without touching the modules via `ingest/ego4d.py` — see
[EGO4D](EGO4D.md).
