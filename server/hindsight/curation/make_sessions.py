"""Pick which sessions to process and write sessions.txt (one line per Slurm array task).

    python -m hindsight.curation.make_sessions --n 30 --per-person 3 --out sessions.txt

Sessions are ranked by how many object-question answer-evidence spans they contain.
Each line: <repo_path>\t<video_id>\t<n_evidence_spans>
"""
import argparse
import collections

from hindsight.data.supermemory import evidence_rows, load_qa, video_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of sessions")
    ap.add_argument("--per-person", type=int, default=1, help="max sessions per participant (spread across people)")
    ap.add_argument("--qa", default=None, help="path to all_qa.json (default: download)")
    ap.add_argument("--out", default="sessions.txt")
    a = ap.parse_args()

    rows = evidence_rows(load_qa(a.qa))
    paths = video_paths()
    counts = collections.Counter((r["video_id"], r["person"]) for r in rows if r["video_id"] in paths)

    chosen, per_person = [], collections.Counter()
    for (vid, person), n in counts.most_common():
        if per_person[person] >= a.per_person:
            continue
        chosen.append((paths[vid], vid, n))
        per_person[person] += 1
        if len(chosen) == a.n:
            break

    with open(a.out, "w") as f:
        for p, vid, n in chosen:
            f.write(f"{p}\t{vid}\t{n}\n")
    print(f"Wrote {len(chosen)} sessions to {a.out}:")
    for p, vid, n in chosen:
        print(f"  {vid}  ({n} evidence spans)")
    print(f"\nSubmit with:  sbatch --array=0-{len(chosen) - 1}%10 pace/scripts/curate_array.sbatch  (from your work dir)")


if __name__ == "__main__":
    main()
