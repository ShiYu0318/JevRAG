"""E3: evidence sufficiency at three granularities.

set          one 4-way choice over the whole passage set
per_passage  P(contains answer) per passage, aggregated (max / noisy-OR / sum),
             thresholded into sufficient vs insufficient
two_stage    aggregated evidence below tau -> insufficient, otherwise the set-level
             choice restricted to sufficient / partial / conflicting

Thresholds are fitted on dev and must be passed in for any other split.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from ..backends.systemone import SystemOneClient
from ..data import tc
from ..metrics import auroc, classwise_ece, confusion, macro_f1, per_class_f1, top_label_ece
from ..ops.sufficiency import aggregate
from ..runlog import Run
from .common import VERDICTS, by_condition, ci, cost_per_1k, merge, passage_grades, verdicts

DEFAULTS: Dict[str, Any] = {
    "dataset": "data/build/jevrag-tc-v0.1.0-k3",
    "conditions": ["S", "I-hard", "I-easy", "P", "C"],
    "extra_conditions": ["S-conj", "J"],
    "n_per_condition": 100,
    "aggregations": ["max", "noisy_or", "sum"],
    "thresholds": None,
    "baselines": [],
    "nli_thresholds": None,
    "lang": "en",
    "workers": 8,
    "seed": 7,
    "bootstrap": 2000,
}


def best_threshold(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Threshold maximising binary F1 of `score >= t`."""
    best, best_f1 = 0.5, -1.0
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, positive) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, positive) if s >= t and not y)
        fn = sum(1 for s, y in zip(scores, positive) if s < t and y)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def _two_stage(evidence: float, tau: float, probs: Dict[str, float]) -> str:
    if evidence < tau:
        return "insufficient"
    rest = {k: probs.get(k, 0.0) for k in ("sufficient", "partial", "conflicting")}
    return max(rest, key=rest.get)


def _report(labels: List[str], preds: List[str], conds: List[str], cfg: Dict[str, Any],
            probs: List[Dict[str, float]] | None = None) -> Dict[str, Any]:
    n = len(labels)
    out: Dict[str, Any] = {
        "macro_f1": ci(n, lambda idx: macro_f1([labels[i] for i in idx], [preds[i] for i in idx], VERDICTS),
                       n=cfg["bootstrap"]),
        "per_class_f1": per_class_f1(labels, preds, VERDICTS),
        "confusion": confusion(labels, preds, VERDICTS),
        "accuracy_by_condition": {c: sum(1 for l, p, k in zip(labels, preds, conds) if k == c and l == p)
                                  / max(1, conds.count(c)) for c in sorted(set(conds))},
    }
    if probs is not None:
        out["top_label_ece"] = top_label_ece(probs, labels)
        out["classwise_ece"] = classwise_ece(probs, labels, VERDICTS)
    return out


def _conjunction_parts(question: str, answer: str) -> List[Tuple[str, str]]:
    try:
        from ..data.build import split_conjunction
    except ImportError:  # builder not available: treat every question as a single question
        return []
    return split_conjunction(question, answer)


def nli_baseline(items: List[dict], labels: List[str], conds: List[str], split: str,
                 cfg: Dict[str, Any]) -> Dict[str, Any]:
    """NLI aggregation baseline. The hypothesis uses the gold answer, so this is optimistic."""
    from ..baselines.nli import NLIScorer, hypothesis, verdict

    scorer = NLIScorer()
    t0 = time.perf_counter()
    feats = []
    for it in items:
        ps = tc.passages(it)
        parts = _conjunction_parts(it["question"], it["answers"][0])
        if parts:
            sub = []
            for q, a in parts:
                sc = scorer.score([(p, hypothesis(q, a)) for p in ps])
                sub.append(max(s["entailment"] for s in sc))
            feats.append(([], [], sub))
        else:
            sc = scorer.score([(p, hypothesis(it["question"], it["answers"][0])) for p in ps])
            feats.append(([s["entailment"] for s in sc], [s["contradiction"] for s in sc], []))
    seconds = time.perf_counter() - t0

    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    th = cfg["nli_thresholds"]
    if th is None:
        if split != "dev":
            raise ValueError("NLI thresholds must come from a dev run for any split other than dev")
        best = max(((te, tcn) for te in grid for tcn in grid),
                   key=lambda t: macro_f1(labels, [verdict(e, c, s, *t) for e, c, s in feats], VERDICTS))
        th = {"tau_e": best[0], "tau_c": best[1]}
    preds = [verdict(e, c, s, th["tau_e"], th["tau_c"]) for e, c, s in feats]
    rep = _report(labels, preds, conds, cfg)
    rep.update({"thresholds": th, "uses_gold_answer": True, "seconds": round(seconds, 1)})
    return rep


def run(client: SystemOneClient, run: Run, split: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = merge(DEFAULTS, overrides)
    conds = list(cfg["conditions"]) + list(cfg["extra_conditions"])
    items = by_condition(tc.load(cfg["dataset"], split, conds), cfg["n_per_condition"], cfg["seed"])
    main = [i for i in items if i["condition"] in cfg["conditions"]]

    # set level
    resp = verdicts(client, items, lang=cfg["lang"], workers=cfg["workers"])
    set_probs = {i["id"]: r["verdict"].probs for i, r in zip(items, resp)}
    set_pred = {i["id"]: r["verdict"].label for i, r in zip(items, resp)}

    # per passage
    evid: Dict[str, Dict[str, float]] = {}
    for it in items:
        rs = passage_grades(client, it["question"], tc.passages(it), lang=cfg["lang"], workers=cfg["workers"],
                            tag={"part": "per_passage", "instance_id": it["id"]})
        ps = [r["ans"].p for r in rs]
        evid[it["id"]] = {a: aggregate(ps, a) for a in cfg["aggregations"]}

    labels = [i["label"] for i in main]
    cnd = [i["condition"] for i in main]
    has_answer = [l != "insufficient" for l in labels]
    thresholds = cfg["thresholds"]
    if thresholds is None:
        if split != "dev":
            raise ValueError("E3 thresholds must come from a dev run for any split other than dev")
        thresholds = {a: best_threshold([evid[i["id"]][a] for i in main], has_answer) for a in cfg["aggregations"]}

    result: Dict[str, Any] = {"n": len(main), "n_by_condition": dict(Counter(cnd)), "thresholds": thresholds}
    result["set"] = _report(labels, [set_pred[i["id"]] for i in main], cnd, cfg, [set_probs[i["id"]] for i in main])
    result["per_passage"] = {}
    for a in cfg["aggregations"]:
        scores = [evid[i["id"]][a] for i in main]
        preds = ["sufficient" if s >= thresholds[a] else "insufficient" for s in scores]
        rep = _report(labels, preds, cnd, cfg)
        rep["evidence_auroc"] = auroc(scores, [int(h) for h in has_answer])
        result["per_passage"][a] = rep
    result["two_stage"] = {
        a: _report(labels, [_two_stage(evid[i["id"]][a], thresholds[a], set_probs[i["id"]]) for i in main], cnd, cfg)
        for a in cfg["aggregations"]}
    majority = Counter(labels).most_common(1)[0][0]
    result["majority"] = _report(labels, [majority] * len(labels), cnd, cfg)

    if "nli" in cfg["baselines"]:
        result["nli"] = nli_baseline(main, labels, cnd, split, cfg)

    conflicts = [set_pred[i["id"]] for i in main if i["condition"] == "C"]
    result["conflict_predicted_as"] = dict(Counter(conflicts))
    result["extra"] = {c: {"n": sum(1 for i in items if i["condition"] == c),
                           "set_accuracy": sum(1 for i in items if i["condition"] == c and set_pred[i["id"]] == i["label"])
                           / max(1, sum(1 for i in items if i["condition"] == c))}
                       for c in cfg["extra_conditions"]}
    result["cost_per_1k_instances"] = cost_per_1k(client, len(items))
    run.summary = {"set_macro_f1": result["set"]["macro_f1"]["value"],
                   **{f"two_stage_{a}_macro_f1": r["macro_f1"]["value"] for a, r in result["two_stage"].items()}}
    run.write("E3.json", result)
    run.write("E3_predictions.json", {i["id"]: {"condition": i["condition"], "label": i["label"],
                                                "set_probs": set_probs[i["id"]], "evidence": evid[i["id"]]}
                                      for i in items})
    return result
