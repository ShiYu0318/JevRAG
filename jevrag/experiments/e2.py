"""E2: calibration audit.

Passage level: P(contains answer) against gold / not gold, split into a hard
group (gold + hard negatives) and an easy group (gold + easy negatives).
Set level: P(sufficient) against the instance label.
Temperature is fitted on a separate split (training by default) and applied to
the evaluation split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..backends.systemone import SystemOneClient
from ..data import tc
from ..metrics import (apply_temperature, aurc, auroc, brier, confident_error_rate, coverage_at_risk, ece,
                       fit_temperature, reliability)
from ..runlog import Run
from .common import by_condition, merge, passage_grades, verdicts

DEFAULTS: Dict[str, Any] = {
    "dataset": "data/build/jevrag-tc-v0.1.0-k3",
    "conditions": ["S", "I-hard", "I-easy"],
    "n_per_condition": 150,
    "fit_split": "training",
    "fit_n_per_condition": 150,
    "n_bins": 15,
    "lang": "en",
    "workers": 8,
    "seed": 7,
}

Pairs = List[Tuple[float, int]]


def collect(client: SystemOneClient, items: List[dict], cfg: Dict[str, Any], part: str) -> Dict[str, Pairs]:
    groups: Dict[str, Pairs] = {"gold": [], "hard_negative": [], "easy_negative": [], "set": []}
    for it in items:
        rs = passage_grades(client, it["question"], tc.passages(it), lang=cfg["lang"], workers=cfg["workers"],
                            tag={"part": f"{part}:passage", "instance_id": it["id"]})
        for c, r in zip(it["contexts"], rs):
            if c["role"] in groups:
                groups[c["role"]].append((r["ans"].p, int(c["role"] == "gold")))
    for it, r in zip(items, verdicts(client, items, lang=cfg["lang"], workers=cfg["workers"], part=f"{part}:set")):
        groups["set"].append((r["verdict"].probs.get("sufficient", 0.0), int(it["label"] == "sufficient")))
    return {
        "hard": groups["gold"] + groups["hard_negative"],
        "easy": groups["gold"] + groups["easy_negative"],
        "all_passages": groups["gold"] + groups["hard_negative"] + groups["easy_negative"],
        "set": groups["set"],
    }


def audit(pairs: Pairs, n_bins: int) -> Dict[str, Any]:
    p = [x for x, _ in pairs]
    y = [t for _, t in pairs]
    # selective view: confidence = max(p, 1-p), correct = thresholded prediction matches
    conf = [max(x, 1 - x) for x in p]
    correct = [int((x >= 0.5) == bool(t)) for x, t in pairs]
    return {
        "n": len(pairs),
        "positive_rate": sum(y) / max(len(y), 1),
        "ece_width": ece(p, y, n_bins, "width"),
        "ece_mass": ece(p, y, n_bins, "mass"),
        "brier": brier(p, y),
        "auroc": auroc(p, y),
        "aurc": aurc(conf, correct),
        "confident_error_rate@0.9": confident_error_rate(conf, correct, 0.9),
        "coverage@risk0.05": coverage_at_risk(conf, correct, 0.05),
        "reliability": [{"conf": c, "acc": a, "n": k} for c, a, k in reliability(p, y, n_bins, "width")],
    }


def run(client: SystemOneClient, run: Run, split: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = merge(DEFAULTS, overrides)
    items = by_condition(tc.load(cfg["dataset"], split, cfg["conditions"]), cfg["n_per_condition"], cfg["seed"])
    ev = collect(client, items, cfg, "eval")
    result: Dict[str, Any] = {"raw": {g: audit(v, cfg["n_bins"]) for g, v in ev.items()}}
    raw = result["raw"]
    result["h2"] = {"ece_easy": raw["easy"]["ece_width"], "ece_hard": raw["hard"]["ece_width"],
                    "gap": raw["hard"]["ece_width"] - raw["easy"]["ece_width"],
                    "supported": raw["easy"]["ece_width"] <= 0.05
                    and raw["hard"]["ece_width"] - raw["easy"]["ece_width"] >= 0.03}

    if cfg["fit_split"]:
        fit_items = by_condition(tc.load(cfg["dataset"], cfg["fit_split"], cfg["conditions"]),
                                 cfg["fit_n_per_condition"], cfg["seed"])
        fit = collect(client, fit_items, cfg, "fit")
        temps = {"passage": fit_temperature(*zip(*fit["all_passages"])), "set": fit_temperature(*zip(*fit["set"]))}
        scaled = {}
        for g, pairs in ev.items():
            t = temps["set" if g == "set" else "passage"]
            p = apply_temperature([x for x, _ in pairs], t)
            scaled[g] = audit(list(zip(p, [y for _, y in pairs])), cfg["n_bins"])
        result["temperature"] = temps
        result["scaled"] = scaled

    run.summary = {"ece_easy": raw["easy"]["ece_width"], "ece_hard": raw["hard"]["ece_width"],
                   "ece_set": raw["set"]["ece_width"]}
    run.write("E2.json", result)
    return result
