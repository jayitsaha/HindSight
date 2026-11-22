from __future__ import annotations

import logging

import numpy as np

from hindsight.video.curation import FrameCurator, Selection, VideoInfo, curation_method

log = logging.getLogger("frame_curation")


class TemporalCurator(FrameCurator):
    @curation_method("temporal")
    def _select_temporal(self, info: VideoInfo, cfg: dict) -> Selection:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        cand = self._sample_indices(info, cfg.get("embed_fps", 1.0))
        n = int(len(cand))
        emb = self.embed_frames(info, cand, cfg["embedder"], cfg.get("cache_dir"))
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

        k = int(cfg["k"])
        linkage = cfg.get("linkage", "ward")

        if k >= n:
            picks, per_frame = [], []
            for c in range(n):
                t = float(cand[c] / info.fps)
                picks.append(int(cand[c]))
                per_frame.append({"cluster": c, "cluster_size": 1,
                                  "cluster_t_min": round(t, 3), "cluster_t_max": round(t, 3)})
            return Selection(indices=np.array(picks), per_frame=per_frame,
                             extra={"k": n, "n_candidates": n, "linkage": linkage,
                                    "embedder": getattr(self._embedder, "name", str(cfg["embedder"]))})

        n_neighbors = int(cfg.get("connectivity_neighbors", 2))
        n_neighbors = max(1, min(n_neighbors, n - 1))
        times = (cand.astype(np.float64) / info.fps).reshape(-1, 1)
        graph = kneighbors_graph(times, n_neighbors=n_neighbors, mode="connectivity", include_self=False)

        model = AgglomerativeClustering(n_clusters=k, connectivity=graph, linkage=linkage)
        labels = model.fit_predict(emb)

        sizes = np.bincount(labels, minlength=k)
        picks, per_frame = [], []
        for c in range(k):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            center = emb[members].mean(0)
            d = np.linalg.norm(emb[members] - center, axis=1)
            m = members[int(d.argmin())]
            picks.append(int(cand[m]))
            span = cand[members] / info.fps
            per_frame.append({"cluster": int(c), "cluster_size": int(sizes[c]),
                              "cluster_t_min": round(float(span.min()), 3),
                              "cluster_t_max": round(float(span.max()), 3)})

        return Selection(indices=np.array(picks), per_frame=per_frame,
                         extra={"k": k, "n_candidates": n, "linkage": linkage,
                                "connectivity_neighbors": n_neighbors,
                                "embedder": getattr(self._embedder, "name", str(cfg["embedder"]))})
