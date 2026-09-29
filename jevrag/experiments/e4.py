"""E4: Traditional vs Simplified script and instruction language.

The same instances are judged in 2 scripts x 2 instruction languages. Flip rate
compares the zh-Hant / en decision with each other condition, paired by
instance. Running this with the english Laya checkpoint is the negative control.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..backends.systemone import SystemOneClient
from ..data import tc
from ..metrics import flip_rate, macro_f1, top_label_ece
from ..runlog import Run
from .common import VERDICTS, by_condition, ci, merge, passage_grades, verdicts

DEFAULTS: Dict[str, Any] = {
    "dataset": "data/build/jevrag-tc-v0.1.0-k3",
    "conditions": ["S", "I-hard", "P", "C"],
    "n_per_condition": 100,
    "passage_level": True,
    "workers": 8,
    "seed": 7,
    "bootstrap": 2000,
}

SETTINGS = [("zh-Hant", "en"), ("zh-Hans", "en"), ("zh-Hant", "zh"), ("zh-Hans", "zh")]


def run(client: SystemOneClient, run: Run, split: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = merge(DEFAULTS, overrides)
    trad = by_condition(tc.load(cfg["dataset"], split, cfg["conditions"]), cfg["n_per_condition"], cfg["seed"])
    wanted = {i["parallel_id"] for i in trad}
    simp = {i["id"]: i for i in tc.load(cfg["dataset"], split, cfg["conditions"], script="zh-Hans")
            if i["id"] in wanted}
    trad = [i for i in trad if i["parallel_id"] in simp]
    sets = {"zh-Hant": trad, "zh-Hans": [simp[i["parallel_id"]] for i in trad]}
    labels = [i["label"] for i in trad]

    decisions: Dict[str, List[str]] = {}
    probs: Dict[str, List[Dict[str, float]]] = {}
    passage: Dict[str, List[bool]] = {}
    passage_p: Dict[str, List[float]] = {}
    for script, lang in SETTINGS:
        key = f"{script}/{lang}"
        rs = verdicts(client, sets[script], lang=lang, workers=cfg["workers"], part=f"e4:{key}")
        decisions[key] = [r["verdict"].label for r in rs]
        probs[key] = [r["verdict"].probs for r in rs]
        if cfg["passage_level"]:
            ps: List[float] = []
            for it in sets[script]:
                ps += [r["ans"].p for r in passage_grades(client, it["question"], tc.passages(it), lang=lang,
                                                          workers=cfg["workers"],
                                                          tag={"part": f"e4:{key}:passage", "instance_id": it["id"]})]
            passage_p[key] = ps
            passage[key] = [p >= 0.5 for p in ps]

    n = len(trad)
    base = "zh-Hant/en"
    result: Dict[str, Any] = {"n": n, "settings": {}}
    for script, lang in SETTINGS:
        key = f"{script}/{lang}"
        d = decisions[key]
        entry: Dict[str, Any] = {
            "macro_f1": macro_f1(labels, d, VERDICTS),
            "top_label_ece": top_label_ece(probs[key], labels),
            "mean_confidence": sum(max(p.values()) for p in probs[key]) / max(n, 1),
            "accuracy": sum(1 for a, b in zip(labels, d) if a == b) / max(n, 1),
        }
        if key != base:
            b = decisions[base]
            entry["flip_rate_vs_base"] = ci(n, lambda idx, d=d: flip_rate([b[i] for i in idx], [d[i] for i in idx]),
                                            n=cfg["bootstrap"])
            flipped = [i for i in range(n) if b[i] != d[i]]
            entry["mean_abs_dp_sufficient_on_flips"] = (
                sum(abs(probs[base][i].get("sufficient", 0) - probs[key][i].get("sufficient", 0)) for i in flipped)
                / len(flipped) if flipped else 0.0)
            if cfg["passage_level"]:
                entry["passage_flip_rate_vs_base"] = flip_rate(passage[base], passage[key])
                entry["passage_mean_abs_dp"] = (sum(abs(x - y) for x, y in zip(passage_p[base], passage_p[key]))
                                                / max(len(passage_p[key]), 1))
        result["settings"][key] = entry
    run.summary = {"script_flip_rate": result["settings"]["zh-Hans/en"]["flip_rate_vs_base"]["value"],
                   "instruction_flip_rate": result["settings"]["zh-Hant/zh"]["flip_rate_vs_base"]["value"]}
    run.write("E4.json", result)
    return result
