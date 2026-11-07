"""Synthetic egocentric day generator.

Simulates a person moving through rooms, picking up and setting down objects,
and doing things — then emits exactly the signals each HINDSIGHT module needs,
with ground truth, so training and evaluation run with no real video or GPU.

The appearance space is built so a *learned* match metric should beat cosine:
each object's vector splits into high-energy "appearance" dims (shared across
look-alike pairs, hit with anisotropic viewpoint noise) and low-energy
"identity" dims (clean, discriminative). Cosine is dominated by appearance and
confuses look-alikes; a scorer that leans on the identity dims pulls them apart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OBJECTS = ["keys", "wallet", "phone", "mug", "glasses", "remote", "book", "bottle", "pen", "charger"]
SCENES = ["kitchen", "living_room", "bedroom", "desk", "bathroom"]
ACTIONS = ["drinking", "reading", "cooking", "washing_hands", "typing",
           "watching_tv", "phone_call", "tidying", "eating", "writing"]

# look-alike pairs share appearance; a trained scorer must use identity dims.
LOOKALIKES = [("keys", "charger"), ("mug", "bottle"), ("phone", "remote"), ("book", "wallet")]

# activities that look alike from a head-mounted camera, so the classifier confuses
# them — keeps action recognition an actual problem rather than ten separable blobs.
SIMILAR_ACTIONS = [("eating", "drinking"), ("typing", "writing"), ("reading", "watching_tv")]

APP_DIM, ID_DIM = 48, 16
EMB_DIM = APP_DIM + ID_DIM
WIN_DIM = 96          # action window feature ("InternVideo2-like")
HO_FEAT = 7           # hand-object features per window
FRAME_W, FRAME_H = 320, 240
FPS = 1.0


@dataclass
class Det:
    frame_idx: int
    t: float
    label: str
    object_id: int
    xyxy: tuple[float, float, float, float]
    emb_idx: int


@dataclass
class PutDown:
    t: float
    frame_idx: int
    object_label: str
    location_id: int
    home_location_id: int
    unusual: bool


@dataclass
class ActionSeg:
    idx: int
    t_start: float
    t_end: float
    label: str
    is_repeat: bool
    repeat_of: int | None
    seg_emb_idx: int


@dataclass
class Query:
    kind: str                 # "object" | "action"
    text: str
    t_asked: float
    target_label: str
    answerable: bool
    answer_frame_idx: int | None
    answer_xyxy: tuple[float, float, float, float] | None
    answer_window: tuple[float, float] | None
    query_emb_idx: int | None


@dataclass
class SyntheticVideo:
    video_id: str
    minutes: float
    objects: list[str]
    scenes: list[str]
    actions_vocab: list[str]
    dets: list[Det]
    putdowns: list[PutDown]
    segments: list[ActionSeg]
    queries: list[Query]
    # dense arrays
    det_emb: np.ndarray            # (N, EMB_DIM) crop embeddings
    query_emb: np.ndarray          # (Qo, EMB_DIM) object-query text embeddings
    seg_emb: np.ndarray            # (S, EMB_DIM) per-segment embeddings
    pd_scene_emb: np.ndarray       # (P, EMB_DIM) scene/context embedding per put-down
    win_feat: np.ndarray           # (W, WIN_DIM) action-window features
    win_label: np.ndarray          # (W,) action class id
    win_t: np.ndarray              # (W,) window center time
    ho_feat: np.ndarray            # (M, HO_FEAT) hand-object window features
    ho_pnr: np.ndarray             # (M,) 1 if window is a put-down PNR
    ho_state: np.ndarray           # (M,) 1 if contact->no-contact state change
    ho_t: np.ndarray               # (M,)
    scene_of_frame: np.ndarray     # (F,) scene id per frame
    n_frames: int = 0

    def save(self, root: str | Path, render: bool = True) -> Path:
        d = Path(root) / self.video_id
        (d / "frames").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "arrays.npz",
            det_emb=self.det_emb, query_emb=self.query_emb, seg_emb=self.seg_emb,
            pd_scene_emb=self.pd_scene_emb,
            win_feat=self.win_feat, win_label=self.win_label, win_t=self.win_t,
            ho_feat=self.ho_feat, ho_pnr=self.ho_pnr, ho_state=self.ho_state, ho_t=self.ho_t,
            scene_of_frame=self.scene_of_frame,
        )
        meta = {
            "video_id": self.video_id, "minutes": self.minutes, "n_frames": self.n_frames,
            "objects": self.objects, "scenes": self.scenes, "actions_vocab": self.actions_vocab,
            "dets": [asdict(x) for x in self.dets],
            "putdowns": [asdict(x) for x in self.putdowns],
            "segments": [asdict(x) for x in self.segments],
            "queries": [asdict(x) for x in self.queries],
        }
        (d / "meta.json").write_text(json.dumps(meta))
        if render:
            _render_frames(self, d / "frames")
        return d


def load(root: str | Path, video_id: str) -> SyntheticVideo:
    d = Path(root) / video_id
    meta = json.loads((d / "meta.json").read_text())
    a = np.load(d / "arrays.npz")
    return SyntheticVideo(
        video_id=meta["video_id"], minutes=meta["minutes"], objects=meta["objects"],
        scenes=meta["scenes"], actions_vocab=meta["actions_vocab"],
        dets=[Det(**x) for x in meta["dets"]],
        putdowns=[PutDown(**x) for x in meta["putdowns"]],
        segments=[ActionSeg(**x) for x in meta["segments"]],
        queries=[Query(**x) for x in meta["queries"]],
        det_emb=a["det_emb"], query_emb=a["query_emb"], seg_emb=a["seg_emb"],
        pd_scene_emb=a["pd_scene_emb"],
        win_feat=a["win_feat"], win_label=a["win_label"], win_t=a["win_t"],
        ho_feat=a["ho_feat"], ho_pnr=a["ho_pnr"], ho_state=a["ho_state"], ho_t=a["ho_t"],
        scene_of_frame=a["scene_of_frame"], n_frames=meta["n_frames"],
    )


class _World:
    """Prototypes, homes and the text (query) projection head."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.home = {o: i % len(SCENES) for i, o in enumerate(OBJECTS)}
        self.proto = {}
        base = {}
        for o in OBJECTS:
            app = rng.standard_normal(APP_DIM)
            idd = rng.standard_normal(ID_DIM) * 0.5
            base[o] = (app, idd)
        for a, b in LOOKALIKES:  # share most appearance energy
            app = (base[a][0] + base[b][0]) / 2
            base[a] = (app + 0.15 * base[a][0], base[a][1])
            base[b] = (app + 0.15 * base[b][0], base[b][1])
        for o in OBJECTS:
            app, idd = base[o]
            app = app / np.linalg.norm(app)
            idd = idd / np.linalg.norm(idd)
            # appearance carries most energy (what cosine sees); identity is weaker
            # but clean and discriminative — the handle a learned metric grabs onto.
            self.proto[o] = np.concatenate([app, 0.55 * idd]).astype(np.float64)
        # modality gap: a small fixed additive offset, not a rotation (keeps alignment).
        self.text_gap = rng.standard_normal(EMB_DIM) * 0.08
        # viewpoint noise: heavy on appearance, light on identity (not zero — else the
        # identity dims are noiseless and a nonlinear scorer hits a suspicious AUC 1.0).
        self.vp_sigma = np.concatenate([np.full(APP_DIM, 0.7), np.full(ID_DIM, 0.09)])
        # one prototype per room; put-down context embeddings are drawn around these,
        # so a scene-clustering model can recognize a room it has seen the object in
        # even at a new spot, which per-location counting cannot.
        self.scene_proto = {i: _unit(rng.standard_normal(EMB_DIM)) for i in range(len(SCENES))}

    def scene_emb(self, scene: int) -> np.ndarray:
        return _unit(self.scene_proto[scene] + self.rng.normal(0, 0.08, EMB_DIM))

    def crop_emb(self, obj: str) -> np.ndarray:
        v = self.proto[obj] + self.rng.normal(0, self.vp_sigma)
        return _unit(v)

    def query_emb(self, obj: str) -> np.ndarray:
        v = self.proto[obj] + self.text_gap + self.rng.normal(0, 0.03, EMB_DIM)
        return _unit(v)


def generate(video_id: str = "synthetic_01", minutes: float = 30.0, seed: int = 7,
             render: bool = True) -> SyntheticVideo:
    rng = np.random.default_rng(seed)
    world = _World(rng)
    n_frames = int(minutes * 60 * FPS)

    scene_of_frame = np.zeros(n_frames, dtype=int)
    cur_scene = 0
    holding: str | None = None
    hold_started = 0

    dets: list[Det] = []
    det_vecs: list[np.ndarray] = []
    putdowns: list[PutDown] = []
    pd_scene_vecs: list[np.ndarray] = []
    ho_rows, ho_pnr, ho_t = [], [], []

    for f in range(n_frames):
        t = f / FPS
        if rng.random() < 0.02:
            cur_scene = int(rng.integers(len(SCENES)))
        scene_of_frame[f] = cur_scene

        visible = [o for o in OBJECTS if world.home[o] == cur_scene]
        if holding and holding not in visible:
            visible.append(holding)

        for o in visible:
            box = _rand_box(rng)
            dets.append(Det(f, t, o, OBJECTS.index(o), box, len(det_vecs)))
            det_vecs.append(world.crop_emb(o))

        # hand-object dynamics: pick up, carry, put down. The features are noisy
        # proxies (grip, hand opening, wrist motion) — deliberately NOT the label,
        # so predicting the release frame stays a real classification problem.
        contact = holding is not None
        speed = float(rng.random() * (0.4 if contact else 0.1))
        openness = float(rng.random() * 0.2)
        wrist = float(rng.random() * 0.15 + speed * 0.3)
        pnr = 0
        if not holding and visible and rng.random() < 0.05:
            holding = visible[int(rng.integers(len(visible)))]
            hold_started = f
        elif holding and (f - hold_started) > 5 and rng.random() < 0.12:
            home = world.home[holding]
            # usually the object goes home; a minority are left in the wrong room
            unusual = rng.random() < 0.25
            put_scene = home if not unusual else int(
                rng.choice([s for s in range(len(SCENES)) if s != home]))
            cur_scene = put_scene
            scene_of_frame[f] = put_scene
            subloc = int(rng.integers(4))                 # spot within the room
            location = put_scene * 4 + subloc
            putdowns.append(PutDown(t, f, holding, location, home * 4, unusual))
            pd_scene_vecs.append(world.scene_emb(put_scene))
            pnr = 1
            holding = None
            quiet = rng.random() < 0.3  # some releases are smooth, with little tell
            boost = 0.25 if quiet else 0.5
            speed = min(1.0, speed + boost * rng.random())
            openness = min(1.0, openness + boost + rng.random() * 0.2)
            wrist = min(1.0, wrist + boost * rng.random())
            if not quiet:
                for j in range(max(0, len(ho_rows) - 2), len(ho_rows)):  # hand opens before release
                    ho_rows[j][5] = min(1.0, ho_rows[j][5] + 0.3 + rng.random() * 0.3)
                    ho_rows[j][1] = min(1.0, ho_rows[j][1] + 0.2)
        elif holding and rng.random() < 0.06:  # fumble / near-drop: looks like a release but isn't
            openness = min(1.0, openness + 0.5)
            wrist = min(1.0, wrist + 0.3)
            if rng.random() < 0.5:
                speed = min(1.0, speed + 0.35)
        grip = float(f - hold_started) if contact else 0.0  # grip that just ended counts at release
        ho_rows.append([float(contact), speed, grip,
                        float(len(visible)), float(cur_scene) / len(SCENES), openness, wrist])
        ho_pnr.append(pnr)
        ho_t.append(t)

    # state-change (object left the hand) is a window-level event around the PNR
    # frame, not the single frame — distinct target, distinct granularity.
    ho_pnr = np.asarray(ho_pnr)
    ho_state = ((ho_pnr + np.roll(ho_pnr, 1) + np.roll(ho_pnr, -1)) > 0).astype(int)

    segments, seg_vecs = _make_actions(rng, world, minutes, scene_of_frame)
    win_feat, win_label, win_t = _make_windows(rng, segments, scene_of_frame)
    queries, query_vecs = _make_queries(rng, world, dets, segments, minutes)

    vid = SyntheticVideo(
        video_id=video_id, minutes=minutes, objects=OBJECTS, scenes=SCENES, actions_vocab=ACTIONS,
        dets=dets, putdowns=putdowns, segments=segments, queries=queries,
        det_emb=_stack(det_vecs), query_emb=_stack(query_vecs), seg_emb=_stack(seg_vecs),
        pd_scene_emb=_stack(pd_scene_vecs),
        win_feat=win_feat, win_label=win_label, win_t=win_t,
        ho_feat=np.asarray(ho_rows, dtype=np.float32), ho_pnr=ho_pnr,
        ho_state=ho_state, ho_t=np.asarray(ho_t, dtype=np.float32),
        scene_of_frame=scene_of_frame, n_frames=n_frames,
    )
    return vid


def _make_actions(rng, world, minutes, scene_of_frame):
    # Each segment gets an instance vector on top of its action/scene appearance.
    # A repeat inherits the instance it repeats (same activity, same sub-goal), so a
    # genuine repeat looks more alike than two coincidental same-label actions — the
    # signal a repetition scorer can use WITHOUT ever seeing the time gap.
    segments, seg_vecs, instances = [], [], []
    t, idx = 0.0, 0
    total = minutes * 60
    recent: list[tuple[int, float, str]] = []  # (idx, t_start, label)
    while t < total - 20:
        dur = float(rng.integers(10, 45))
        repeat_of, is_repeat, label = None, False, None
        candidates = [r for r in recent if t - r[1] < 600]
        if candidates and rng.random() < 0.35:
            repeat_of, _, label = candidates[int(rng.integers(len(candidates)))]
            is_repeat = True
        else:
            label = ACTIONS[int(rng.integers(len(ACTIONS)))]
        scene = int(scene_of_frame[min(int(t), len(scene_of_frame) - 1)])
        if is_repeat:
            inst = instances[repeat_of] + rng.normal(0, 0.2, EMB_DIM)
        else:
            inst = rng.standard_normal(EMB_DIM)
        instances.append(inst)
        appearance = _unit(_action_vec(label, scene))
        seg_vecs.append(_unit(0.7 * appearance + 0.7 * _unit(inst)))
        segments.append(ActionSeg(idx, t, t + dur, label, is_repeat, repeat_of, idx))
        recent.append((idx, t, label))
        recent = recent[-12:]
        t += dur + float(rng.integers(2, 18))
        idx += 1
    return segments, seg_vecs


def _make_windows(rng, segments, scene_of_frame, win_s: float = 2.0):
    # The action identity dominates; the room it happens in is a nuisance direction
    # plus noise. Same action across rooms must still classify together, so accuracy
    # is non-trivial rather than a perfectly separable ten-centroid toy.
    partner = {a: b for a, b in SIMILAR_ACTIONS} | {b: a for a, b in SIMILAR_ACTIONS}
    feats, labels, times = [], [], []
    for s in segments:
        action = _action_vec(s.label, 0, WIN_DIM)
        if s.label in partner:  # bleed a confusable activity into its twin
            shared = _action_vec("pair:" + "+".join(sorted([s.label, partner[s.label]])), 0, WIN_DIM)
            action = 0.6 * shared + 0.4 * action
        tt = s.t_start
        while tt < s.t_end:
            scene = int(scene_of_frame[min(int(tt), len(scene_of_frame) - 1)])
            nuisance = _action_vec("room", scene, WIN_DIM)
            feats.append(action + 0.4 * nuisance + rng.normal(0, 0.9, WIN_DIM))
            labels.append(ACTIONS.index(s.label))
            times.append(tt + win_s / 2)
            tt += win_s
    return (np.asarray(feats, dtype=np.float32), np.asarray(labels), np.asarray(times, dtype=np.float32))


def _make_queries(rng, world, dets, segments, minutes):
    queries, qvecs = [], []
    by_label: dict[str, list[Det]] = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)
    total = minutes * 60
    first_seen = {o: min(d.t for d in ds) for o, ds in by_label.items()}
    # answerable "where is my X?" — ask after the object has been around
    for _ in range(16):
        obj = OBJECTS[int(rng.integers(len(OBJECTS)))]
        ds = by_label.get(obj, [])
        if not ds:
            continue
        t_ask = float(rng.uniform(max(first_seen[obj] + 30, total * 0.3), total))
        prior = [d for d in ds if d.t <= t_ask]
        if not prior:
            continue
        last = max(prior, key=lambda d: d.t)
        queries.append(Query("object", f"where is my {obj}?", t_ask, obj, True,
                             last.frame_idx, last.xyxy, (last.t - 1, last.t), len(qvecs)))
        qvecs.append(world.query_emb(obj))
    # unanswerable "where is my X?" — asked before the object ever appears
    askable = [o for o in OBJECTS if o in first_seen and first_seen[o] > 90]
    for obj in rng.permutation(askable)[:5]:
        t_ask = float(rng.uniform(10, first_seen[obj] - 5))
        queries.append(Query("object", f"where is my {obj}?", t_ask, obj, False,
                             None, None, None, len(qvecs)))
        qvecs.append(world.query_emb(obj))
    # action queries — "when did I last X?"
    for _ in range(6):
        t_ask = float(rng.uniform(total * 0.5, total))
        act = ACTIONS[int(rng.integers(len(ACTIONS)))]
        prior = [s for s in segments if s.t_end <= t_ask and s.label == act]
        if prior:
            last = max(prior, key=lambda s: s.t_end)
            queries.append(Query("action", f"when did I last {act}?", t_ask, act, True,
                                 None, None, (last.t_start, last.t_end), None))
        else:
            queries.append(Query("action", f"when did I last {act}?", t_ask, act, False,
                                 None, None, None, None))
    return queries, qvecs


def _action_vec(name: str, scene: int, dim: int = EMB_DIM) -> np.ndarray:
    # stable across processes (unlike hash()), so train/val prototypes match even
    # when the two splits are generated in separate runs.
    digest = hashlib.md5(f"{name}:{scene}:{dim}".encode()).hexdigest()[:8]
    return np.random.default_rng(int(digest, 16)).standard_normal(dim)


def _rand_box(rng) -> tuple[float, float, float, float]:
    w = rng.uniform(30, 90); h = rng.uniform(30, 90)
    x0 = rng.uniform(0, FRAME_W - w); y0 = rng.uniform(0, FRAME_H - h)
    return (float(x0), float(y0), float(x0 + w), float(y0 + h))


def _render_frames(vid: SyntheticVideo, out: Path) -> None:
    by_frame: dict[int, list[Det]] = {}
    for d in vid.dets:
        by_frame.setdefault(d.frame_idx, []).append(d)
    scene_bg = [(28, 32, 40), (40, 30, 30), (30, 38, 30), (36, 36, 44), (26, 36, 40)]
    for f in range(vid.n_frames):
        img = Image.new("RGB", (FRAME_W, FRAME_H), scene_bg[int(vid.scene_of_frame[f])])
        draw = ImageDraw.Draw(img)
        for d in by_frame.get(f, []):
            c = _obj_color(d.label)
            draw.rectangle(d.xyxy, fill=c, outline=(230, 230, 230))
        img.save(out / f"{f:06d}.png")


def _obj_color(label: str) -> tuple[int, int, int]:
    h = abs(hash(label))
    return (60 + h % 180, 60 + (h // 180) % 180, 60 + (h // 32400) % 180)


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / max(np.linalg.norm(v), 1e-8)).astype(np.float32)


def _stack(vecs: list[np.ndarray]) -> np.ndarray:
    return np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, EMB_DIM), np.float32)
