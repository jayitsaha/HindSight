"""
hindsight.video.curation: pick a subset of frames from a video and write them out as a new video.
(Also usable as a standalone file, `frame_curation.py`; it only imports third-party packages.)
Full guide: docs/FRAME_CURATION.md. Default config: server/configs/frame_curation.json.

Methods (choose with config "method", the `method=` argument, or `--method`):
  * "uniform": sample frames at a fixed rate (for example 1 or 3 frames per second).
  * "kmeans":  embed frames with a CLIP-style model, run KMeans with K clusters, keep the
               frame nearest each cluster centre, and stitch those frames in time order.

Every method only *selects* source-frame indices. Reading, writing and the JSON manifest
are shared, so adding a method means writing one selector function:

    class MyCurator(FrameCurator):
        @curation_method("my_method")
        def _select_my_method(self, info, cfg):
            ...
            return Selection(indices=np.array([...]), extra={...})

Usage
-----
    python -m hindsight.video.curation --write-default-config config.json
    python -m hindsight.video.curation in.mp4 out.mp4 --config configs/frame_curation.json
    python -m hindsight.video.curation in.mp4 out.mp4 --method kmeans --set kmeans.k=16
    python -m hindsight.video.curation in.mp4 out.mp4 --method uniform --set uniform.fps=3

    from hindsight.video.curation import FrameCurator
    cur = FrameCurator("configs/frame_curation.json")
    res = cur.curate("in.mp4", "out_uniform.mp4", method="uniform")
    res = cur.curate("in.mp4", "out_kmeans.mp4", method="kmeans", k=24)

Each run also writes <output>.json listing, for every output frame, the source frame index
and its timestamp (and, for kmeans, the cluster and its size).

Requirements: numpy, opencv-python(-headless), scikit-learn; for kmeans also torch plus
open_clip_torch (backend "open_clip") or transformers (backend "hf").
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x=None, **_):
        return x

log = logging.getLogger("frame_curation")

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "uniform",
    "io": {
        "start_s": None,          # only consider frames at or after this time (seconds)
        "end_s": None,            # only consider frames before this time, e.g. the question time
        "output_fps": None,       # playback fps of the output video; null = method default
        "resize": None,           # [width, height] for the output video; null = keep source size
        "rotate": 0,              # 0, 90, 180, 270 (clockwise), applied before embedding and writing
        "overlay_timestamp": False,  # draw the source timestamp on each output frame
        "codec": "mp4v",          # OpenCV fourcc
        "reencode_h264": False,   # re-encode with ffmpeg/libx264 so browsers can play it (needs ffmpeg)
        "write_manifest": True,   # write <output>.json with selected frames and timestamps
    },
    "uniform": {
        "fps": 1.0,               # frames per second to sample; output plays at this fps by default
    },
    "kmeans": {
        "k": 32,                  # number of clusters = number of output frames
        "embed_fps": None,        # null = embed every frame (as specified); e.g. 1.0 = embed 1 frame per second (much faster)
        "algorithm": "kmeans",    # "kmeans" or "minibatch" (faster on very many frames)
        "n_init": 3,
        "random_state": 0,
        "normalize": True,        # L2-normalise embeddings (cosine geometry)
        "output_fps": 1.0,        # default playback fps for the stitched centroid video
        "cache_dir": None,        # directory to cache embeddings (.npz); null = no cache
        "embedder": {
            "backend": "open_clip",           # "open_clip" or "hf"
            "model": "ViT-B-32",              # open_clip: model name; hf: repo id, e.g. "openai/clip-vit-base-patch32"
            "pretrained": "laion2b_s34b_b79k",  # open_clip only
            "device": "auto",                 # "auto", "cuda", "mps", "cpu"
            "batch_size": 64,
        },
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# --------------------------------------------------------------------------------------
# Video helpers
# --------------------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:  # some containers don't report a frame count; count by grabbing
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    if fps <= 0:
        raise ValueError(f"Video reports no fps: {path}")
    return VideoInfo(str(path), float(fps), n, w, h)


def iter_frames(path: str, indices: Sequence[int]) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) for the requested indices, in increasing order.
    Skipped frames are only grabbed, not decoded into images, which keeps sequential reads fast."""
    wanted = np.unique(np.asarray(indices, dtype=int))
    if wanted.size == 0:
        return
    cap = cv2.VideoCapture(str(path))
    try:
        cur, j = 0, 0
        last = int(wanted[-1])
        while cur <= last and j < len(wanted):
            if not cap.grab():
                break
            if cur == wanted[j]:
                ok, frame = cap.retrieve()
                if ok:
                    yield cur, frame
                j += 1
            cur += 1
    finally:
        cap.release()


def rotate_frame(frame: np.ndarray, deg: int) -> np.ndarray:
    deg = int(deg) % 360
    if deg == 0:
        return frame
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(deg)
    if code is None:
        raise ValueError("rotate must be 0, 90, 180 or 270")
    return cv2.rotate(frame, code)


def fmt_ts(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------------------
# Embedders (swappable CLIP-style image encoders)
# --------------------------------------------------------------------------------------
def _pick_device(device: str):
    import torch
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenClipEmbedder:
    def __init__(self, model: str, pretrained: Optional[str], device: str = "auto", batch_size: int = 64, **_):
        import open_clip
        import torch
        self.torch = torch
        self.device = _pick_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.eval().to(self.device)
        self.batch_size = batch_size
        self.name = f"open_clip:{model}:{pretrained}"

    def embed(self, images: List["Image.Image"]) -> np.ndarray:
        torch = self.torch
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
            e = self.model.encode_image(x)
        return e.float().cpu().numpy()


class HFEmbedder:
    """Any Hugging Face model with get_image_features (CLIP, SigLIP, SigLIP 2, ...)."""

    def __init__(self, model: str, device: str = "auto", batch_size: int = 64, **_):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = _pick_device(device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval().to(self.device)
        self.batch_size = batch_size
        self.name = f"hf:{model}"

    def embed(self, images: List["Image.Image"]) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
            e = self.model.get_image_features(**inputs)
        return e.float().cpu().numpy()


EMBEDDERS: Dict[str, Callable[..., Any]] = {"open_clip": OpenClipEmbedder, "hf": HFEmbedder}


def build_embedder(cfg: dict):
    cfg = dict(cfg)
    backend = cfg.pop("backend", "open_clip")
    if backend not in EMBEDDERS:
        raise ValueError(f"Unknown embedder backend {backend!r}; options: {list(EMBEDDERS)}")
    return EMBEDDERS[backend](**cfg)


# --------------------------------------------------------------------------------------
# Curator
# --------------------------------------------------------------------------------------
@dataclass
class Selection:
    indices: np.ndarray                        # source frame indices, sorted ascending
    extra: Dict[str, Any] = field(default_factory=dict)   # per-method info for the manifest
    per_frame: Optional[List[Dict[str, Any]]] = None      # optional per-selected-frame info


@dataclass
class CurationResult:
    method: str
    input: str
    output: str
    manifest: Optional[str]
    indices: np.ndarray
    timestamps: np.ndarray
    output_fps: float
    seconds: float
    extra: Dict[str, Any]


def curation_method(name: str):
    """Decorator that registers a selector method on FrameCurator (or a subclass)."""
    def deco(fn):
        fn._curation_name = name
        return fn
    return deco


class FrameCurator:
    """Frame curation: choose which frames of a video to keep and write them as a new video."""

    def __init__(self, config: Optional[dict | str | os.PathLike] = None, **overrides):
        user = {}
        if isinstance(config, (str, os.PathLike)):
            with open(config) as f:
                user = json.load(f)
        elif isinstance(config, dict):
            user = config
        self.config = deep_merge(DEFAULT_CONFIG, user)
        for k, v in overrides.items():          # e.g. FrameCurator(cfg, **{"kmeans.k": 16})
            set_dotted(self.config, k, v)
        self._embedder = None
        self._embedder_key = None

    # ---- registry ----
    @classmethod
    def methods(cls) -> Dict[str, Callable]:
        out = {}
        for klass in reversed(cls.__mro__):
            for attr in vars(klass).values():
                name = getattr(attr, "_curation_name", None)
                if name:
                    out[name] = attr
        return out

    # ---- public API ----
    def curate(self, input_path: str, output_path: str, method: Optional[str] = None, **params) -> CurationResult:
        """Run one curation method and write the output video (plus manifest).
        `params` override that method's config section, e.g. curate(..., method="kmeans", k=16)."""
        t0 = time.time()
        method = method or self.config["method"]
        registry = self.methods()
        if method not in registry:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(registry)}")
        cfg = deep_merge(self.config.get(method, {}), params)
        io = self.config["io"]

        info = probe_video(input_path)
        log.info("Input %s: %.2f fps, %d frames (%.1f min), %dx%d",
                 input_path, info.fps, info.n_frames, info.duration / 60, info.width, info.height)

        sel: Selection = registry[method](self, info, cfg)
        idx = np.asarray(sel.indices, dtype=int)
        if idx.size == 0:
            raise RuntimeError("Selection is empty (check start_s/end_s and method params)")
        order = np.argsort(idx)
        idx = idx[order]
        per_frame = [sel.per_frame[i] for i in order] if sel.per_frame else None
        ts = idx / info.fps

        out_fps = io.get("output_fps") or self._default_output_fps(method, cfg)
        self._write_video(info, idx, output_path, out_fps)

        manifest = None
        if io.get("write_manifest", True):
            manifest = str(Path(output_path).with_suffix(".json"))
            frames = []
            for i, (fi, t) in enumerate(zip(idx.tolist(), ts.tolist())):
                row = {"out_index": i, "src_index": fi, "t_sec": round(t, 3), "t_str": fmt_ts(t)}
                if per_frame:
                    row.update(per_frame[i])
                frames.append(row)
            with open(manifest, "w") as f:
                json.dump({"method": method, "method_config": cfg, "io": io,
                           "source": {"path": info.path, "fps": info.fps, "n_frames": info.n_frames,
                                      "width": info.width, "height": info.height, "duration_s": info.duration},
                           "output": {"path": str(output_path), "fps": out_fps, "n_frames": int(len(idx))},
                           "extra": sel.extra, "frames": frames}, f, indent=2, default=_json_default)

        secs = time.time() - t0
        log.info("Wrote %s: %d frames at %.2f fps (%.1fs)", output_path, len(idx), out_fps, secs)
        return CurationResult(method, str(input_path), str(output_path), manifest, idx, ts, out_fps, secs, sel.extra)

    # ---- shared helpers ----
    def _candidate_range(self, info: VideoInfo) -> Tuple[int, int]:
        io = self.config["io"]
        lo = 0 if io.get("start_s") is None else max(0, int(np.ceil(io["start_s"] * info.fps)))
        hi = info.n_frames if io.get("end_s") is None else min(info.n_frames, int(np.floor(io["end_s"] * info.fps)))
        if hi <= lo:
            raise ValueError(f"Empty frame range: start_s={io.get('start_s')} end_s={io.get('end_s')}")
        return lo, hi

    def _sample_indices(self, info: VideoInfo, fps: Optional[float]) -> np.ndarray:
        """Indices at `fps` frames per second inside the candidate range (all frames if fps is None or >= source fps)."""
        lo, hi = self._candidate_range(info)
        if fps is None or fps >= info.fps:
            return np.arange(lo, hi)
        if fps <= 0:
            raise ValueError("fps must be > 0")
        times = np.arange(lo / info.fps, hi / info.fps, 1.0 / fps)
        idx = np.round(times * info.fps).astype(int)
        return np.unique(idx[(idx >= lo) & (idx < hi)])

    def _default_output_fps(self, method: str, cfg: dict) -> float:
        if method == "uniform":
            return float(cfg["fps"])               # same duration as the source
        return float(cfg.get("output_fps") or 1.0)

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        return rotate_frame(frame, self.config["io"].get("rotate", 0))

    def _write_video(self, info: VideoInfo, indices: np.ndarray, output_path: str, fps: float) -> None:
        io = self.config["io"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path if not io.get("reencode_h264") else str(Path(output_path).with_suffix(".tmp.mp4"))
        writer = None
        written = 0
        for fi, frame in tqdm(iter_frames(info.path, indices), total=len(indices), desc="write", leave=False):
            frame = self._prep(frame)
            if io.get("resize"):
                frame = cv2.resize(frame, tuple(int(v) for v in io["resize"]), interpolation=cv2.INTER_AREA)
            if io.get("overlay_timestamp"):
                txt = fmt_ts(fi / info.fps)
                scale = max(0.5, frame.shape[0] / 720)
                cv2.putText(frame, txt, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale))
                cv2.putText(frame, txt, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(2 * scale)))
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*io.get("codec", "mp4v")), fps, (w, h))
                if not writer.isOpened():
                    raise IOError(f"Could not open VideoWriter for {tmp} (codec {io.get('codec')})")
            writer.write(frame)
            written += 1
        if writer is not None:
            writer.release()
        if written != len(indices):
            log.warning("Wrote %d of %d requested frames (video shorter than reported?)", written, len(indices))
        if io.get("reencode_h264"):
            if not shutil.which("ffmpeg"):
                raise RuntimeError("reencode_h264=true needs ffmpeg on PATH")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-r", str(fps), output_path], check=True)
            os.remove(tmp)

    def get_embedder(self, emb_cfg: dict):
        key = json.dumps(emb_cfg, sort_keys=True)
        if self._embedder is None or self._embedder_key != key:
            self._embedder = build_embedder(emb_cfg)
            self._embedder_key = key
        return self._embedder

    def embed_frames(self, info: VideoInfo, indices: np.ndarray, emb_cfg: dict,
                     cache_dir: Optional[str] = None) -> np.ndarray:
        """Embed the given frames with the configured image encoder. Returns (len(indices), D)."""
        from PIL import Image

        cache_file = None
        if cache_dir:
            st = os.stat(info.path)
            key = hashlib.md5(json.dumps([os.path.abspath(info.path), st.st_size, int(st.st_mtime),
                                          emb_cfg, self.config["io"].get("rotate", 0),
                                          indices.tolist()]).encode()).hexdigest()[:16]
            cache_file = Path(cache_dir) / f"{Path(info.path).stem}_{key}.npz"
            if cache_file.exists():
                log.info("Loading cached embeddings %s", cache_file)
                return np.load(cache_file)["emb"]

        if len(indices) > 20000:
            log.warning("Embedding %d frames; consider a lower embed_fps", len(indices))
        model = self.get_embedder(emb_cfg)
        bs = int(emb_cfg.get("batch_size", 64))
        out, batch = [], []
        for _, frame in tqdm(iter_frames(info.path, indices), total=len(indices), desc="embed", leave=False):
            frame = self._prep(frame)
            batch.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(batch) == bs:
                out.append(model.embed(batch)); batch = []
        if batch:
            out.append(model.embed(batch))
        emb = np.concatenate(out).astype(np.float32)
        if len(emb) != len(indices):
            raise RuntimeError(f"Embedded {len(emb)} frames but expected {len(indices)}")
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, emb=emb, indices=indices)
        return emb

    # ---- curation methods ----
    @curation_method("uniform")
    def _select_uniform(self, info: VideoInfo, cfg: dict) -> Selection:
        idx = self._sample_indices(info, float(cfg["fps"]))
        return Selection(indices=idx, extra={"sample_fps": float(cfg["fps"])})

    @curation_method("kmeans")
    def _select_kmeans(self, info: VideoInfo, cfg: dict) -> Selection:
        from sklearn.cluster import KMeans, MiniBatchKMeans

        cand = self._sample_indices(info, cfg.get("embed_fps"))
        emb = self.embed_frames(info, cand, cfg["embedder"], cfg.get("cache_dir"))
        if cfg.get("normalize", True):
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

        k = int(cfg["k"])
        if k > len(cand):
            log.warning("k=%d > %d candidate frames; using k=%d", k, len(cand), len(cand))
            k = len(cand)
        Algo = MiniBatchKMeans if cfg.get("algorithm") == "minibatch" else KMeans
        km = Algo(n_clusters=k, n_init=cfg.get("n_init", 3), random_state=cfg.get("random_state", 0)).fit(emb)

        sizes = np.bincount(km.labels_, minlength=k)
        picks, per_frame = [], []
        for c in range(k):
            members = np.where(km.labels_ == c)[0]
            if members.size == 0:
                continue
            d = np.linalg.norm(emb[members] - km.cluster_centers_[c], axis=1)
            m = members[d.argmin()]
            picks.append(int(cand[m]))
            span = cand[members] / info.fps
            per_frame.append({"cluster": c, "cluster_size": int(sizes[c]),
                              "cluster_t_min": round(float(span.min()), 3),
                              "cluster_t_max": round(float(span.max()), 3),
                              "dist_to_centroid": round(float(d.min()), 4)})
        return Selection(indices=np.array(picks), per_frame=per_frame,
                         extra={"k": k, "n_candidates": int(len(cand)), "embed_fps": cfg.get("embed_fps"),
                                "embedder": getattr(self._embedder, "name", str(cfg["embedder"])),
                                "inertia": float(km.inertia_)})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_value(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Frame curation: uniform fps sampling or CLIP+KMeans keyframes.")
    p.add_argument("input", nargs="?", help="input video")
    p.add_argument("output", nargs="?", help="output video (.mp4)")
    p.add_argument("--config", help="JSON config (missing keys use defaults)")
    p.add_argument("--method", help=f"curation method; available: {sorted(FrameCurator.methods())}")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set kmeans.k=16 --set uniform.fps=3 --set io.end_s=600")
    p.add_argument("--write-default-config", metavar="PATH", help="write the default config to PATH and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if a.write_default_config:
        with open(a.write_default_config, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print("Wrote", a.write_default_config)
        return
    if not (a.input and a.output):
        p.error("input and output are required")

    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        overrides[k] = _parse_value(v)
    cur = FrameCurator(a.config, **overrides)
    res = cur.curate(a.input, a.output, method=a.method)
    print(json.dumps({"method": res.method, "output": res.output, "manifest": res.manifest,
                      "n_frames": int(len(res.indices)), "output_fps": res.output_fps,
                      "seconds": round(res.seconds, 1)}, indent=2))


if __name__ == "__main__":
    main()
