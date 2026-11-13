"""Pick which sessions to process and write sessions.txt (one line per Slurm array task).

    python -m hindsight.curation.make_sessions --n 30 --per-person 3 --out sessions.txt
    python -m hindsight.curation.make_sessions --all --skip-done out --out sessions_rest.txt   # everything not yet curated

Sessions are ranked by how many object-question answer-evidence spans they contain. --all takes every session that
an Object & Location question needs (answer evidence or the session the question is asked in) and that exists in the
repo; --skip-done drops sessions whose outputs for every --ks value are already in that folder.
Each line: <repo_path>\t<video_id>\t<n_evidence_spans>
"""
import argparse
import collections
import os

from hindsight.data.supermemory import SKILL, evidence_rows, load_qa, video_paths


def is_done(out_dir, vid, ks):
    """True when out_dir/<vid>/ has the KMeans and uniform video + manifest for every K."""
    return all(os.path.isfile(os.path.join(out_dir, vid, f"{m}_K{k}.{ext}"))
               for k in ks for m in ("kmeans", "uniform") for ext in ("mp4", "json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of sessions")
    ap.add_argument("--per-person", type=int, default=1, help="max sessions per participant (spread across people)")
    ap.add_argument("--qa", default=None, help="path to all_qa.json (default: download)")
    ap.add_argument("--out", default="sessions.txt")
    ap.add_argument("--all", action="store_true", help="every needed session (ignores --n / --per-person)")
    ap.add_argument("--skip-done", default=None, metavar="OUT_DIR", help="skip sessions already curated in OUT_DIR")
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 100, 150], help="K values a session must have to count as done")
    a = ap.parse_args()

    qa = load_qa(a.qa)
    rows = evidence_rows(qa)
    paths = video_paths()
    counts = collections.Counter((r["video_id"], r["person"]) for r in rows if r["video_id"] in paths)
    if a.all:  # sessions only needed as the question's own session (unanswerable questions) count too, with 0 spans
        for r in qa:
            vid = r["metadata"]["primary_video_id"]
            if r["metadata"]["skill"] == SKILL and vid in paths:
                counts[(vid, r["subject"])] += 0
        missing = sorted({e["video_id"] for r in qa if r["metadata"]["skill"] == SKILL
                          for e in r["answer_evidence"]["evidence_list"]} - set(paths))
        if missing:
            print(f"Not in the repo (questions needing these cannot be served): {', '.join(missing)}")

    chosen, per_person, skipped = [], collections.Counter(), 0
    for (vid, person), n in counts.most_common():
        if a.skip_done and is_done(a.skip_done, vid, a.ks):
            skipped += 1
            continue
        if not a.all and per_person[person] >= a.per_person:
            continue
        chosen.append((paths[vid], vid, n))
        per_person[person] += 1
        if not a.all and len(chosen) == a.n:
            break

    with open(a.out, "w") as f:
        for p, vid, n in chosen:
            f.write(f"{p}\t{vid}\t{n}\n")
    if a.skip_done:
        print(f"Skipped {skipped} sessions already curated in {a.skip_done}/")
    print(f"Wrote {len(chosen)} sessions to {a.out}:")
    for p, vid, n in chosen:
        print(f"  {vid}  ({n} evidence spans)")
    print(f"\nSubmit with:  sbatch --array=0-{len(chosen) - 1}%10 pace/scripts/curate_array.sbatch  (from your work dir)")


if __name__ == "__main__":
    main()
