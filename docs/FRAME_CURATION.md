# Frame Curation (`hindsight.video.curation`)

> Code: `server/hindsight/video/curation.py` · default config: `server/configs/frame_curation.json` · tests: `server/tests/test_curation.py`.
> Run the commands below from `server/` (or after `pip install -e server`). The file has no project imports, so it also works copied standalone as `frame_curation.py` (e.g. on Colab).

Pick a small set of frames from a long video and write them out as a new, shorter video, along with a JSON file giving each frame's original timestamp.
It's the first stage of the HindSight pipeline: its output videos are the input to the VLM stage.

| Method | What it does | Uses the question? |
|---|---|---|
| `uniform` | Samples frames at a fixed rate (for example 3 fps, 1 fps, 0.1 fps) | No |
| `kmeans` | Embeds frames with a CLIP-style model, runs KMeans with K clusters, keeps the frame nearest each cluster centre, and stitches them in time order | No |

Every method only *selects frame indices*. Reading the video, writing the output and the JSON manifest are shared, so new methods are easy to add (see [Adding a new method](#adding-a-new-method)).

---

## Install

```bash
pip install numpy opencv-python-headless scikit-learn tqdm pillow
pip install torch open_clip_torch          # for method "kmeans" (backend "open_clip")
pip install transformers                   # only if you use embedder backend "hf"
```

Optional: `ffmpeg` on your PATH if you set `io.reencode_h264: true`.

---

## Quick start

### Command line

```bash
# write the default config (then edit it)
python -m hindsight.video.curation --write-default-config my_config.json

# uniform sampling at 1 frame per second
python -m hindsight.video.curation input.mp4 out_uniform.mp4 --config configs/frame_curation.json --method uniform --set uniform.fps=1

# CLIP + KMeans, 32 keyframes
python -m hindsight.video.curation input.mp4 out_kmeans.mp4 --config configs/frame_curation.json --method kmeans --set kmeans.k=32
```

- `--config` is optional. Any key you leave out uses the default.
- `--method` overrides `"method"` in the config.
- `--set KEY=VALUE` overrides any config value using dotted paths and can be repeated. Values are parsed as JSON, so `--set io.resize=[448,448]`, `--set io.end_s=600` and `--set io.overlay_timestamp=true` all work.
- `-v` gives more detailed logging.

### Python

```python
from hindsight.video.curation import FrameCurator

cur = FrameCurator("configs/frame_curation.json")              # or a dict, or nothing for all defaults

res = cur.curate("input.mp4", "out_uniform.mp4", method="uniform", fps=3)
res = cur.curate("input.mp4", "out_kmeans.mp4",  method="kmeans",  k=32)

print(res.indices)      # source frame indices that were kept (sorted)
print(res.timestamps)   # their times in seconds
print(res.manifest)     # path to the JSON manifest
```

Swapping methods is just the `method=` argument; the rest of the call stays the same.

---

## Three ways to set parameters

They are applied in this order, and later ones win:

1. **Defaults** built into the file (`DEFAULT_CONFIG`).
2. **Config file or dict** passed to `FrameCurator(...)`. You only need the keys you want to change.
3. **Overrides:**
   - At construction, with dotted keys. Use this for **any** section, including `io`:
     ```python
     cur = FrameCurator("configs/frame_curation.json", **{"io.resize": [448, 448], "kmeans.k": 16})
     ```
   - Per call, as keyword arguments to `curate()`. These override **only the chosen method's section**:
     ```python
     cur.curate("in.mp4", "out.mp4", method="kmeans", k=64, embed_fps=1.0)
     cur.curate("in.mp4", "out.mp4", method="kmeans",
                embedder={"backend": "hf", "model": "google/siglip2-base-patch16-224"})
     ```
     To change `io` settings (resize, time window and so on) between calls, edit `cur.config["io"][...]` or create a new `FrameCurator`.
   - On the command line: `--set section.key=value`.

---

## Config reference

Full default config (`server/configs/frame_curation.json`):

```json
{
  "method": "uniform",
  "io": {
    "start_s": null,
    "end_s": null,
    "output_fps": null,
    "resize": null,
    "rotate": 0,
    "overlay_timestamp": false,
    "codec": "mp4v",
    "reencode_h264": false,
    "write_manifest": true
  },
  "uniform": {
    "fps": 1.0
  },
  "kmeans": {
    "k": 32,
    "embed_fps": null,
    "algorithm": "kmeans",
    "n_init": 3,
    "random_state": 0,
    "normalize": true,
    "output_fps": 1.0,
    "cache_dir": null,
    "embedder": {
      "backend": "open_clip",
      "model": "ViT-B-32",
      "pretrained": "laion2b_s34b_b79k",
      "device": "auto",
      "batch_size": 64
    }
  }
}
```

### `method`
Which method runs when none is given in the call or on the command line: `"uniform"` or `"kmeans"` (or any method you add).

### `io`: shared by all methods

| Key | Default | Meaning |
|---|---|---|
| `start_s` | `null` | Only consider frames at or after this time (seconds). |
| `end_s` | `null` | Only consider frames before this time. **Set this to the question time** so no frames from after the question are used. |
| `output_fps` | `null` | Playback fps of the output video. `null` uses the method's default (uniform: same as `uniform.fps`, so the output lasts as long as the original; kmeans: `kmeans.output_fps`). |
| `resize` | `null` | `[width, height]` of output frames. `null` keeps the source size (1408×1408 for Aria). `[448, 448]` makes files about 10× smaller. |
| `rotate` | `0` | Rotate frames clockwise by 0/90/180/270 degrees, before embedding **and** writing. |
| `overlay_timestamp` | `false` | Draw the source timestamp (`MM:SS`) on each output frame. Useful when the VLM needs to reason about time. |
| `codec` | `"mp4v"` | OpenCV fourcc. `mp4v` plays in QuickTime, VLC and Python, but not in every browser. |
| `reencode_h264` | `false` | Re-encode the output to H.264 with ffmpeg so it plays everywhere, including Colab and browsers. Needs `ffmpeg`. |
| `write_manifest` | `true` | Write `<output>.json` next to the video (see below). |

### `uniform`

| Key | Default | Meaning |
|---|---|---|
| `fps` | `1.0` | Frames per second to keep. Values at or above the source fps keep every frame. `0.0339` gives about 32 frames from a 15.7-minute video. |

### `kmeans`

| Key | Default | Meaning |
|---|---|---|
| `k` | `32` | Number of clusters, which is the number of output frames. If K is more than the number of candidate frames, it is reduced to match. |
| `embed_fps` | `null` | Which frames are clustered. `null` embeds **every** frame (28k frames for a 15.7-min, 30-fps video). `1.0` embeds 1 frame/s: about 30× faster with nearly the same result. **Recommended: `1.0`.** |
| `algorithm` | `"kmeans"` | `"kmeans"` (scikit-learn KMeans) or `"minibatch"` (MiniBatchKMeans, faster on very many frames). |
| `n_init` | `3` | KMeans restarts. The best of these is kept. |
| `random_state` | `0` | Seed, for reproducible clusters. |
| `normalize` | `true` | L2-normalise embeddings before clustering, which effectively clusters by cosine similarity. |
| `output_fps` | `1.0` | Default playback fps of the keyframe video, so each keyframe shows for 1 second. |
| `cache_dir` | `null` | Folder for cached embeddings (`.npz`). With a cache, rerunning with a different `k` skips embedding. The cache is tied to the video file, the embedder settings, `rotate` and `embed_fps`. |
| `embedder` | see below | Which image model to use. |

#### `kmeans.embedder`: swapping the CLIP model

| Key | Meaning |
|---|---|
| `backend` | `"open_clip"` or `"hf"` (Hugging Face transformers). |
| `model` | open_clip: model name (`"ViT-B-32"`, `"ViT-L-14"`, …). hf: repo id (`"openai/clip-vit-base-patch32"`, `"google/siglip2-base-patch16-224"`, …). |
| `pretrained` | open_clip only: weights tag (`"laion2b_s34b_b79k"`, `"openai"`, …). |
| `device` | `"auto"` (CUDA → Apple MPS → CPU), or `"cuda"`, `"mps"`, `"cpu"`. |
| `batch_size` | Frames per forward pass. Lower it if you run out of GPU memory. |

Examples:

```json
"embedder": {"backend": "open_clip", "model": "ViT-L-14", "pretrained": "openai", "device": "auto", "batch_size": 64}
```
```json
"embedder": {"backend": "hf", "model": "google/siglip2-base-patch16-224", "device": "auto", "batch_size": 64}
```

Frames are always resized to the model's input size (224×224 for ViT-B/32) for embedding only. The output video is not affected.

---

## Outputs

Each run writes two files:

1. **`<output>.mp4`**: the selected frames in time order.
2. **`<output>.json`**: the manifest:

```json
{
  "method": "kmeans",
  "method_config": { "k": 32, "embed_fps": 1.0, "...": "..." },
  "io": { "...": "..." },
  "source": { "path": "in.mp4", "fps": 30.0, "n_frames": 28333, "width": 1408, "height": 1408, "duration_s": 944.4 },
  "output": { "path": "out_kmeans.mp4", "fps": 1.0, "n_frames": 32 },
  "extra":  { "k": 32, "n_candidates": 945, "embedder": "open_clip:ViT-B-32:laion2b_s34b_b79k", "inertia": 90.97 },
  "frames": [
    { "out_index": 0, "src_index": 1230, "t_sec": 41.0, "t_str": "00:41",
      "cluster": 5, "cluster_size": 37, "cluster_t_min": 22.0, "cluster_t_max": 75.0, "dist_to_centroid": 0.31 }
  ]
}
```

- `out_index`: position in the output video.
- `src_index` / `t_sec` / `t_str`: frame number and time in the **original** video. Use these to match frames against evidence time spans or to give the VLM real timestamps.
- KMeans only: `cluster`, `cluster_size` (how many candidate frames that keyframe stands for), and `cluster_t_min`/`cluster_t_max` (the time range the cluster covers).

`curate()` also returns a `CurationResult` with `method`, `input`, `output`, `manifest`, `indices`, `timestamps`, `output_fps`, `seconds` (run time) and `extra`.

---

## Common recipes

**Frames for the VLM stage (smaller files, timestamps drawn on, plays everywhere):**
```bash
python -m hindsight.video.curation in.mp4 out.mp4 --config configs/frame_curation.json --method kmeans \
  --set kmeans.k=32 --set kmeans.embed_fps=1 --set kmeans.cache_dir=cache \
  --set io.resize=[448,448] --set io.overlay_timestamp=true --set io.reencode_h264=true
```

**Only use video before the question (no peeking at the future):**
```bash
python -m hindsight.video.curation in.mp4 out.mp4 --method kmeans --set io.end_s=1234.0
```

**Several K values for one video. Embedding runs once, then the cache is reused:**
```python
cur = FrameCurator("configs/frame_curation.json", **{"kmeans.cache_dir": "cache", "kmeans.embed_fps": 1.0})
for k in [8, 16, 32, 64]:
    cur.curate("in.mp4", f"out_k{k}.mp4", method="kmeans", k=k)
```

**Uniform with the same frame budget as KMeans (for a fair comparison):**
```python
from hindsight.video.curation import probe_video
info = probe_video("in.mp4")
cur.curate("in.mp4", "out_uniform_k32.mp4", method="uniform", fps=32 / info.duration)
```

**Many videos:**
```python
cur = FrameCurator("configs/frame_curation.json")
for vid in ["a.mp4", "b.mp4"]:
    cur.curate(vid, f"out/{vid[:-4]}_kmeans.mp4", method="kmeans")   # the model loads once and is reused
```

---

## Adding a new method

Write one function that returns the frame indices to keep, and register it with `@curation_method`.
Reading, writing, the manifest, `start_s`/`end_s`, resize and so on all work automatically.

```python
import numpy as np
from hindsight.video.curation import FrameCurator, Selection, curation_method

class MyCurator(FrameCurator):
    @curation_method("clip_topk")
    def _select_clip_topk(self, info, cfg):
        cand = self._sample_indices(info, cfg.get("embed_fps", 1.0))      # candidate frames (respects start_s/end_s)
        emb = self.embed_frames(info, cand, cfg["embedder"], cfg.get("cache_dir"))
        # ... score frames, choose `keep` (positions into cand) ...
        keep = np.arange(min(cfg["k"], len(cand)))
        return Selection(indices=cand[keep],
                         per_frame=[{"score": 0.0} for _ in keep],   # optional: extra columns in the manifest
                         extra={"note": "anything for the manifest"})

cur = MyCurator({"clip_topk": {"k": 32, "embed_fps": 1.0, "embedder": {"backend": "open_clip",
                 "model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}}})
cur.curate("in.mp4", "out_topk.mp4", method="clip_topk")
```

Helpers available inside a method:
- `self._sample_indices(info, fps)`: frame indices at a given rate within `start_s`/`end_s` (`None` = all frames).
- `self.embed_frames(info, indices, embedder_cfg, cache_dir)`: (N, D) embeddings, cached if `cache_dir` is set.
- `self.get_embedder(embedder_cfg)`: the loaded model, reused across calls.
- `info`: a `VideoInfo` with `path`, `fps`, `n_frames`, `width`, `height` and `duration`.

`FrameCurator.methods()` lists every registered method.

---

## Performance notes

- **Reading the video is the main cost.** The whole video is read even when only a few frames are kept. On PACE (L40S), a 15.7-min Aria video took **~44 s (uniform 1 fps)** and **~135 s (KMeans, `embed_fps=1`)**.
- **`embed_fps: null` (every frame)** is ~30× more embedding work. Use `1.0` unless you really need every frame.
- **Use `cache_dir`** whenever you'll rerun the same video with a different `k` or method.
- **GPU:** any NVIDIA GPU works, and CLIP ViT-B/32 is small. On PACE, request a type explicitly, for example `--gres=gpu:l40s:1`. **Avoid `mi210` and `r9700` (AMD)**: standard PyTorch builds only support NVIDIA, so the run would silently use the CPU.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Output video won't play in a browser or Colab | Set `io.reencode_h264: true` (needs ffmpeg). |
| Frames look sideways | Set `io.rotate` to 90, 180 or 270. |
| `CUDA available: False` / very slow | You're on the CPU. Check that you're on a GPU node, and use a CUDA build of PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`). |
| Out of GPU memory | Lower `kmeans.embedder.batch_size`. |
| `Empty frame range` | `start_s`/`end_s` exclude every frame; check the times are in seconds from the start of the video. |
| `ModuleNotFoundError: hindsight` | Run from `server/`, `pip install -e server`, or `export PYTHONPATH=<repo>/server`. |
