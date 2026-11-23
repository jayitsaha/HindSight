"""Print a side-by-side table of metrics.json from several evaluation output dirs."""

import json
import os
import sys

ROWS = [
    ("QA accuracy", "accuracy"), ("Score w/ partial credit", "score_with_partial_credit"),
    ("Answerable accuracy", "answerable_accuracy"), ("Abstention accuracy", "abstention_accuracy"),
    ("Answerability F1 (answerable)", "answerability_f1"), ("Answerability macro-F1", "answerability_macro_f1"),
    ("Hallucination rate", "hallucination_rate"), ("Over-abstention rate", "over_abstention_rate"),
    ("Vague-option rate", "vague_rate"), ("Invalid rate", "invalid_rate"),
]


def main(dirs):
    metrics = {os.path.basename(d.rstrip("/")): json.load(open(os.path.join(d, "metrics.json"))) for d in dirs}
    names = list(metrics)
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    print("| n | " + " | ".join(str(metrics[n]["total_samples"]) for n in names) + " |")
    for label, key in ROWS:
        print(f"| {label} | " + " | ".join(f"{metrics[n][key] * 100:.1f}%" for n in names) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
