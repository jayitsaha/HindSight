# Ego4D for HINDSIGHT

We use the **episodic memory** benchmarks (VQ2D, MQ, NLQ) and the
**hands-and-objects** PNR annotations, plus precomputed video features so no
GPU video backbone is needed. Loaders live in `hindsight.ingest.ego4d`; the
synthetic generator (`ingest/synthetic.py`) mirrors these structures, so
everything trained on synthetic data has a real-data counterpart here.

## Getting the data

Sign the license at https://ego4ddataset.com to get AWS credentials, then
follow https://ego4d-data.org/docs/start-here/#download-data:

```bash
pip install ego4d awscli
aws configure --profile ego4d          # paste the credentials from the license email

export EGO4D_ROOT=/path/to/ego4d

# annotations for all benchmarks (small, start here)        ~3 GB
ego4d --output_directory "$EGO4D_ROOT" --datasets annotations --version v2

# 5-min canonical clips, restricted to the benchmarks we use  ~200-700 GB
ego4d --output_directory "$EGO4D_ROOT" --datasets clips --version v2 \
      --benchmarks vq nlq mq fho

# precomputed per-clip video features (Omnivore Swin-L)       ~100 GB
ego4d --output_directory "$EGO4D_ROOT" --datasets omnivore_video_swinl --version v2

# optional: VQ2D-specific visual features
ego4d --output_directory "$EGO4D_ROOT" --datasets vq2d_features --version v2
```

Sizes are approximate; the CLI prints the exact total and asks for
confirmation before downloading. Annotations alone are enough to run the
loaders and tests.

## Resulting layout

```
$EGO4D_ROOT/
  ego4d.json                          # video metadata manifest
  v2/
    annotations/
      vq_{train,val,test_unannotated}.json
      moments_{train,val,test_unannotated}.json
      nlq_{train,val,test_unannotated}.json
      fho_oscc-pnr_{train,val,test_unannotated}.json
      fho_hands_{train,val,test_unannotated}.json
    clips/                            # {clip_uid}.mp4
    omnivore_video_swinl/             # {video_uid}.pt, (T, 1536) tensors
    vq2d_features/
```

## Pointing the loaders at it

```python
from hindsight.ingest import ego4d

paths = ego4d.Ego4DPaths.from_env()            # reads EGO4D_ROOT, or pass a root
vq   = ego4d.load_vq2d(paths.vq2d("val"))      # list[VisualQuery]
mq   = ego4d.load_mq(paths.mq("val"))          # list[MomentSegment], 110 labels
nlq  = ego4d.load_nlq(paths.nlq("val"))        # list[NLQuery]
pnr  = ego4d.load_pnr(paths.pnr("val"))        # list[PNRClip]; paths.hands() also works
feat = ego4d.read_features(paths.features(), video_uid)   # (T, D) float32
```

Every loader raises `FileNotFoundError` with the exact `ego4d` command to run
when a file is missing.

## What the loaders return

- `VisualQuery` — the query object crop (`crop_frame`, `crop_xyxy`) and the
  response track; `response_frame`/`response_xyxy` are the **last appearance**
  (max frame in the track), which is what object retrieval needs. Maps onto
  the synthetic `Query(kind="object")` with `answer_frame_idx`/`answer_xyxy`.
- `MomentSegment` — `(label, t_start, t_end)` per action instance,
  clip-relative seconds. Maps onto synthetic `ActionSeg`.
- `NLQuery` — question text + `(t_start, t_end)` answer window. Maps onto
  synthetic `Query(kind="action")` with `answer_window`.
- `PNRClip` — `pnr_frame` plus `pre_frame`/`post_frame` and the
  `state_change` flag. Maps onto the synthetic `ho_pnr`/`ho_state` labels.

## Schema assumptions (parsed defensively)

The v2 JSONs share a `videos -> clips -> annotations` nesting; benchmarks
differ below that. Where the schema varies across releases we accept both
spellings and skip malformed entries instead of crashing:

- **Boxes** come as `{x, y, width, height}`; we convert to `xyxy` to match
  the rest of the codebase. Frame ids may be `frame_number` or `frame`.
- **MQ** segments live in `annotations[].labels[]` with
  `start_time`/`end_time`; we also accept `moment_start_sec`/`moment_end_sec`.
- **NLQ** answer windows are `clip_start_sec`/`clip_end_sec` *inside each
  language query* (the clip-level keys with the same names are the clip's
  position in the full video — not used). Test-split queries without a window
  are skipped.
- **PNR**: `fho_oscc-pnr` is flat (one entry per clip, int `pnr_frame`,
  bool `state_change`); `fho_hands` nests `frames[]` per action with
  `pnr_frame`/`pre_frame`/`post_frame` as `{"frame": ...}` dicts. `load_pnr`
  handles both; `state_change` defaults to true when absent (oscc positives).
- **VQ2D** `query_sets` is a dict keyed `"1".."3"`; entries with
  `is_valid: false` or an empty `response_track` are dropped.
- **Features** are one tensor per `video_uid` (`torch.load`-able `.pt`;
  `.npy` also accepted), returned as `(T, D)` float32. Omnivore Swin-L is
  D=1536 at ~1.87 features/sec; InternVideo2 features follow the same layout
  under their own directory name.
