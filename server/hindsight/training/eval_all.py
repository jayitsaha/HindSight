"""Train every HINDSIGHT module on one split, evaluate on another, report.

Orchestration only — each model/metric lives in its own module and is imported
here. `run()` fits the four modules on train_01, scores them on val_01, writes
the fitted models and a report.json under settings.model_dir, and returns the
numbers as a nested dict. Runnable as `python -m hindsight.training.eval_all`.
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from sklearn.cluster import KMeans

from ..action import ActionClassifier, RepetitionScorer, SameClassRule
from ..action.repetition import is_repeat_pair
from ..assistant import Assistant
from ..config import settings
from ..metrics import (ari, auroc, coverage_at_accuracy, macro_f1, mrr, nmi,
                       recall_at_k, temporal_map)
from ..putdown import CountingBaseline, MisplacementModel, PNRDetector
from ..retrieval import KMeansIndex, build_scorer
from ..retrieval.service import _training_pairs
from .datasets import load

_SCORERS = ("cosine", "logreg", "rf", "mlp")


def run(train_id: str = "train_01", val_id: str = "val_01",
        minutes: float = 30.0, save: bool = True, verbose: bool = True) -> dict:
    train = load(train_id, minutes)
    val = load(val_id, minutes)

    report = {
        "retrieval": _retrieval(train, val, save),
        "putdown": _putdown(train, val, save),
        "action": _action(train, val, save),
        "assistant": _assistant(train, val, save),
        "clustering": _clustering(val),
    }
    if save:
        (settings.model_dir / "report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        _print_table(report, train_id, val_id)
    return report


# --- retrieval ------------------------------------------------------------
def _answerable_object(video):
    return [q for q in video.queries if q.kind == "object" and q.answerable]


def _score_retrieval(scorer, val) -> dict:
    C = val.det_emb
    labels = np.array([d.label for d in val.dets])
    ranked, ys, ss = [], [], []
    for q in _answerable_object(val):
        qv = val.query_emb[q.query_emb_idx]
        s = scorer.rank(qv, C)
        order = np.argsort(-s)
        ranked.append(labels[order] == q.target_label)
        ys.append((labels == q.target_label).astype(int))
        ss.append(s)
    return {
        "recall@1": recall_at_k(ranked, 1),
        "mrr": mrr(ranked),
        "auc": auroc(np.concatenate(ss), np.concatenate(ys)),
    }


def _retrieval(train, val, save) -> dict:
    X, y = _training_pairs(train, np.random.default_rng(0))
    fitted, scorers = {}, {}
    for name in _SCORERS:
        sc = build_scorer(name).fit(X, y)
        scorers[name] = sc
        fitted[name] = _score_retrieval(sc, val)

    # approximate nearest-neighbor index vs brute force on cosine crop match
    C = val.det_emb
    labels = np.array([d.label for d in val.dets])
    index = KMeansIndex(n_clusters=32).build(C)
    bf, ix = [], []
    for q in _answerable_object(val):
        qv = val.query_emb[q.query_emb_idx]
        bf.append(labels[int(np.argmax(C @ qv))] == q.target_label)
        ix.append(labels[int(index.query(qv, topk=1)[0])] == q.target_label)

    if save:
        joblib.dump({n: scorers[n] for n in ("logreg", "rf", "mlp")},
                    settings.model_dir / "retrieval_scorers.joblib")
    return {
        "scorers": fitted,
        "kmeans_index": {"recall@1": float(np.mean(ix)),
                         "brute_recall@1": float(np.mean(bf))},
    }


# --- put-down -------------------------------------------------------------
def _putdown(train, val, save) -> dict:
    pnr = PNRDetector().fit(train)
    pnr_prob, state_prob = pnr.predict(val.ho_feat)
    state_acc = float(((state_prob >= 0.5).astype(int) == val.ho_state).mean())

    model = MisplacementModel().fit(train)
    base = CountingBaseline().fit(train)
    y = np.array([int(p.unusual) for p in val.putdowns])
    model_s = [model.score(p.object_label, e) for p, e in zip(val.putdowns, val.pd_scene_emb)]
    base_s = [base.score(p.object_label, p.location_id) for p in val.putdowns]

    if save:
        pnr.save(settings.model_dir / "pnr.pkl")
        # defaultdict(lambda) isn't picklable; freeze to plain dicts before dumping
        model.counts = dict(model.counts)
        base.counts = {k: dict(v) for k, v in base.counts.items()}
        joblib.dump(model, settings.model_dir / "misplacement.joblib")
    return {
        "pnr": {"auroc": auroc(pnr_prob, val.ho_pnr), "state_acc": state_acc},
        "misplacement": {"model_auroc": auroc(model_s, y),
                         "baseline_auroc": auroc(base_s, y)},
    }


# --- action ---------------------------------------------------------------
def _repeat_pairs(video, scorer, same_label_only: bool):
    emb = video.seg_emb
    y, p = [], []
    for i, a in enumerate(video.segments):
        for b in video.segments[i + 1:]:
            if b.t_start - a.t_start > settings.repeat_window_s:
                continue
            if same_label_only and a.label != b.label:
                continue
            y.append(int(is_repeat_pair(a, b)))
            p.append(scorer.score(a, b, emb[a.seg_emb_idx], emb[b.seg_emb_idx]))
    return np.array(y), np.array(p)


def _windows_to_segments(times, labels, scores, win_s: float = 2.0):
    order = np.argsort(times)
    segs, cur, start, end, ss = [], None, 0.0, 0.0, []
    for i in order:
        t, lab = float(times[i]), int(labels[i])
        if lab != cur or (ss and t - end > win_s * 1.5):
            if cur is not None:
                segs.append((start, end, cur, float(np.mean(ss))))
            cur, start, ss = lab, t - win_s / 2, []
        end = t + win_s / 2
        ss.append(float(scores[i]))
    if cur is not None:
        segs.append((start, end, cur, float(np.mean(ss))))
    return segs


def _action(train, val, save) -> dict:
    clf = ActionClassifier().fit(train)
    labels, probs = clf.predict_windows(val.win_feat)
    window_acc = float((labels == val.win_label).mean())

    # MQ-style temporal localization: stitch predicted windows into segments
    preds = _windows_to_segments(val.win_t, labels, probs.max(1))
    vocab = val.actions_vocab
    gts = [(s.t_start, s.t_end, vocab.index(s.label), 1.0) for s in val.segments]
    tmap = temporal_map(preds, gts)

    rep = RepetitionScorer().fit(train)
    rule = SameClassRule()
    # the discriminative task is repeats vs coincidental same-label pairs
    y, rep_s = _repeat_pairs(val, rep, same_label_only=True)
    _, rule_s = _repeat_pairs(val, rule, same_label_only=True)

    if save:
        clf.save(settings.model_dir / "action.joblib")
        joblib.dump(rep, settings.model_dir / "repetition.joblib")
    return {
        "classifier": {"macro_f1": macro_f1(val.win_label, labels), "window_acc": window_acc},
        "localization": {"mAP": tmap["mAP"]},
        "repetition": {"scorer_auroc": auroc(rep_s, y), "rule_auroc": auroc(rule_s, y),
                       "n_pos": int(y.sum()), "n_pairs": int(len(y))},
    }


def _clustering(val) -> dict:
    # Unsupervised identity/type discovery, scored against true labels. Object
    # crops cluster poorly from raw embeddings — viewpoint/appearance noise buries
    # the identity dims — which is exactly why retrieval needs a learned metric.
    # Action windows, where the activity signal dominates, cluster cleanly. (HDBSCAN
    # is exercised separately on resting scenes in putdown/misplacement.)
    obj_true = np.array([d.object_id for d in val.dets])
    obj_pred = KMeans(n_clusters=len(np.unique(obj_true)), n_init=10,
                      random_state=0).fit_predict(val.det_emb)
    act_true = val.win_label
    act_pred = KMeans(n_clusters=len(np.unique(act_true)), n_init=10,
                      random_state=0).fit_predict(val.win_feat)
    return {
        "object_identity": {"nmi": nmi(obj_true, obj_pred), "ari": ari(obj_true, obj_pred)},
        "action_type": {"nmi": nmi(act_true, act_pred), "ari": ari(act_true, act_pred)},
    }


# --- assistant ------------------------------------------------------------
def _assistant(train, val, save) -> dict:
    asst = Assistant(scorer_name="logreg").fit(train)
    # judge_queries reports whether committing would actually be RIGHT (label/window),
    # not merely that we didn't refuse — so a confident wrong answer counts as wrong.
    rows = asst.judge_queries(val)
    conf = np.array([r["conf"] for r in rows])
    correct = np.array([r["correct"] for r in rows], bool)
    unans = ~np.array([r["answerable"] for r in rows], bool)
    answered = ~np.array([r["refused"] for r in rows], bool)

    rng = np.random.default_rng(0)
    if save:
        joblib.dump(asst, settings.model_dir / "assistant.joblib")
    return {
        "coverage_at_80": coverage_at_accuracy(conf, correct, 0.80),
        "aurc_model": asst.gate.aurc(conf, correct),
        "aurc_random": asst.gate.aurc(rng.random(len(conf)), correct),
        "refusal_recall": float(((~answered) & unans).sum() / max(unans.sum(), 1)),
        "coverage": float(answered.mean()),
        "accuracy_answered": float(correct[answered].mean()) if answered.any() else float("nan"),
        "n_unanswerable": int(unans.sum()),
        "n_queries": int(len(rows)),
    }


# --- console report -------------------------------------------------------
def _row(name: str, *cells: str) -> str:
    return f"  {name:<26}" + "".join(f"{c:>12}" for c in cells)


def _print_table(r: dict, train_id: str, val_id: str) -> None:
    print(f"\nHINDSIGHT eval  —  train {train_id} -> val {val_id}")
    print("=" * 64)

    print("\nRETRIEVAL   object query -> correct crop")
    print(_row("scorer", "Recall@1", "MRR", "AUC"))
    for name in _SCORERS:
        m = r["retrieval"]["scorers"][name]
        print(_row(name, f"{m['recall@1']:.3f}", f"{m['mrr']:.3f}", f"{m['auc']:.3f}"))
    km = r["retrieval"]["kmeans_index"]
    print(_row("kmeans-index R@1", f"{km['recall@1']:.3f}", f"(bf {km['brute_recall@1']:.3f})"))

    print("\nPUT-DOWN")
    pd = r["putdown"]
    print(_row("PNR AUROC", f"{pd['pnr']['auroc']:.3f}"))
    print(_row("state-change acc", f"{pd['pnr']['state_acc']:.3f}"))
    print(_row("misplacement AUROC", f"{pd['misplacement']['model_auroc']:.3f}",
               f"(count {pd['misplacement']['baseline_auroc']:.3f})"))

    print("\nACTION")
    ac = r["action"]
    print(_row("macro-F1", f"{ac['classifier']['macro_f1']:.3f}"))
    print(_row("per-window acc", f"{ac['classifier']['window_acc']:.3f}"))
    print(_row("localization mAP", f"{ac['localization']['mAP']:.3f}"))
    rep = ac["repetition"]
    print(_row("repetition AUROC", f"{rep['scorer_auroc']:.3f}",
               f"(rule {rep['rule_auroc']:.3f}, {rep['n_pos']}/{rep['n_pairs']} pos)"))

    print("\nCLUSTERING   unsupervised, vs true labels")
    cl = r["clustering"]
    print(_row("object identity NMI/ARI", f"{cl['object_identity']['nmi']:.3f}",
               f"{cl['object_identity']['ari']:.3f}"))
    print(_row("action type NMI/ARI", f"{cl['action_type']['nmi']:.3f}",
               f"{cl['action_type']['ari']:.3f}"))

    print("\nASSISTANT   calibrated refusal")
    a = r["assistant"]
    print(_row("coverage @ 80% acc", f"{a['coverage_at_80']:.3f}"))
    print(_row("AURC model/random", f"{a['aurc_model']:.4f}", f"{a['aurc_random']:.4f}"))
    print(_row("refusal recall (unans)", f"{a['refusal_recall']:.3f}", f"(n={a['n_unanswerable']})"))
    print(_row("coverage / acc answered", f"{a['coverage']:.3f}", f"{a['accuracy_answered']:.3f}"))
    print("=" * 64)


if __name__ == "__main__":
    run()
