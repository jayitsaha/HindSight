# Notebooks (frame curation)

| Notebook | What it does |
|---|---|
| `hindsight_kmeans_midterm.ipynb` | Colab: download SuperMemory-VQA sessions, 1-fps frames → CLIP → PCA → KMeans sweep; evidence coverage vs frame budget (KMeans vs uniform vs random), elbow/silhouette, NMI vs rooms, t-SNE, timelines, cluster gallery. |
| `test_frame_curation_colab.ipynb` | Colab: run `FrameCurator` (uniform + KMeans) on one session, play the outputs, keyframe grids, evidence hits. |

Both run on a Colab T4 with an `HF_TOKEN` secret. The test notebook needs the curation code: upload
`server/hindsight/video/curation.py` renamed to `frame_curation.py`, or put `server/` on `sys.path`.

The large-scale version of this experiment (30 sessions on PACE) is `hindsight.curation` + `pace/scripts/curate_array.sbatch`;
results are in `results/curation/`. Guide: `docs/FRAME_CURATION.md`.
