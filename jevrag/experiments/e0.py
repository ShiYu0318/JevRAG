"""E0: backend profile. Latency, length sensitivity, packed vs separate,
consistency, tokens per character."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from ..backends.systemone import SystemOneClient, choice, noul
from ..data import tc
from ..metrics import accuracy, flip_rate, percentile
from ..ops.grade import grade_passages
from ..questions import TEXT, sufficiency_v1
from ..runlog import Run

DEFAULTS: Dict[str, Any] = {
    "dataset": "data/build/jevrag-tc-v0.1.0-k3",
    "length_dataset": "data/build/jevrag-tc-v0.1.0-k10",
    "latency": {"n_questions": [1, 4, 16, 64], "state_chars": [128, 384, 1000, 4000, 8000], "reps": 30},
    "length": {"n_passages": [1, 3, 5, 10], "n_items": 200},
    "packed": {"n_items": 100},
    "consistency": {"n_items": 200, "repeats": 5},
    "seed": 7,
}

# Two rewordings of sufficiency.v1 used only to measure decision flips.
PARAPHRASES = [
    ("Do these passages, read together, contain what is needed to answer the question?",
     {"sufficient": "Everything needed to answer is present",
      "partial": "Some of what is needed is present, some is missing",
      "conflicting": "The passages give incompatible answers",
      "insufficient": "The answer is not in the passages"}),
    ("Judge the evidence: is it enough to answer the question?",
     {"sufficient": "Enough evidence for a full answer",
      "partial": "Evidence covers only part of the answer",
      "conflicting": "Evidence contradicts itself",
      "insufficient": "No evidence for the answer"}),
]


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _verdict(r) -> str:
    return r["verdict"].label


def latency(client: SystemOneClient, cfg: Dict[str, Any], filler: str) -> List[Dict[str, Any]]:
    rows = []
    for chars in cfg["state_chars"]:
        text = (filler * (chars // max(len(filler), 1) + 1))[:chars]
        for nq in cfg["n_questions"]:
            qs = {f"q{i}": noul(f"Does the text mention item {i}?") for i in range(nq)}
            ms, tokens = [], []
            for rep in range(cfg["reps"]):
                r = client.ask({"text": text, "rep": rep}, qs, use_cache=False,
                               tag={"part": "latency", "chars": chars, "n_questions": nq})
                ms.append(r.latency_ms)
                tokens.append(int(r.usage.get("input_tokens") or r.usage.get("prompt_tokens") or 0))
            rows.append({"state_chars": chars, "n_questions": nq, "p50_ms": percentile(ms, 50),
                         "p95_ms": percentile(ms, 95), "input_tokens_mean": sum(tokens) / len(tokens)})
    return rows


def _truncate(inst: dict, n: int) -> List[str]:
    ctx = inst["contexts"]
    gold = [c for c in ctx if c["role"] == "gold"]
    hard = sorted((c for c in ctx if c["role"] == "hard_negative"), key=lambda c: c["retrieval_rank"] or 0)
    chosen = (gold[:1] + hard[:n - 1]) if inst["condition"] == "S" else hard[:n]
    return [c["text"] for c in chosen]


def length_curve(client: SystemOneClient, cfg: Dict[str, Any], items: List[dict]) -> List[Dict[str, Any]]:
    rows = []
    for n in cfg["n_passages"]:
        want, got, chars = [], [], []
        for inst in items:
            ps = _truncate(inst, n)
            if len(ps) < n:
                continue
            r = client.ask(*sufficiency_v1(inst["question"], ps),
                           tag={"part": "length", "instance_id": inst["id"], "n_passages": n})
            want.append(inst["label"] == "sufficient")
            got.append(_verdict(r) == "sufficient")
            chars.append(sum(len(p) for p in ps))
        rows.append({"n_passages": n, "n": len(want), "accuracy": accuracy(want, got),
                     "state_chars_median": percentile(chars, 50)})
    return rows


def packed_vs_separate(client: SystemOneClient, items: List[dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels: Dict[str, List[bool]] = {}
    for mode in ("pointwise", "packed"):
        top1, ms, answers = [], [], []
        for inst in items:
            ps = tc.passages(inst)
            t0 = time.perf_counter()
            g = grade_passages(client, inst["question"], ps, mode=mode, with_safety=False)
            ms.append((time.perf_counter() - t0) * 1000)
            gold = next(i for i, c in enumerate(inst["contexts"]) if c["role"] == "gold")
            top1.append(max(g, key=lambda x: (x.answers_p, x.relevance)).index == gold)
            answers.extend(x.answers_p >= 0.5 for x in g)
        labels[mode] = answers
        out[mode] = {"gold_top1": sum(top1) / max(len(top1), 1), "p50_ms": percentile(ms, 50),
                     "p95_ms": percentile(ms, 95)}
    out["answer_label_agreement"] = 1 - flip_rate(labels["pointwise"], labels["packed"])
    return out


def consistency(client: SystemOneClient, cfg: Dict[str, Any], items: List[dict]) -> Dict[str, Any]:
    base = [_verdict(client.ask(*sufficiency_v1(i["question"], tc.passages(i)))) for i in items]
    repeat_flips = []
    for rep in range(cfg["repeats"]):
        again = [_verdict(client.ask(*sufficiency_v1(i["question"], tc.passages(i)), use_cache=False,
                                     tag={"part": "repeat", "rep": rep, "instance_id": i["id"]}))
                 for i in items]
        repeat_flips.append(flip_rate(base, again))
    para = {}
    for j, (instr, opts) in enumerate(PARAPHRASES):
        got = []
        for i in items:
            state = {"question": i["question"], "passages": tc.passages(i)}
            qs = {"verdict": choice(instr, opts), "false_premise": noul(TEXT["en"]["false_premise"])}
            got.append(_verdict(client.ask(state, qs, tag={"part": "paraphrase", "variant": j,
                                                          "instance_id": i["id"]})))
        para[f"paraphrase_{j}"] = flip_rate(base, got)
    return {"repeat_flip_rate_max": max(repeat_flips, default=0.0), "repeat_flip_rates": repeat_flips,
            "paraphrase_flip_rates": para, "n": len(items)}


def run(client: SystemOneClient, run: Run, split: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _merge(DEFAULTS, overrides)
    seed = cfg["seed"]
    main = tc.load(cfg["dataset"], split, ["S", "I-hard"])
    filler = "".join(c["text"] for c in main[0]["contexts"]) if main else "測試文字。"
    result: Dict[str, Any] = {}

    result["latency"] = latency(client, cfg["latency"], filler)
    try:
        long_items = tc.load(cfg["length_dataset"], split, ["S", "I-hard"], limit=cfg["length"]["n_items"], seed=seed)
        result["length"] = length_curve(client, cfg["length"], long_items)
    except FileNotFoundError:
        result["length"] = f"skipped: {cfg['length_dataset']} not built"
    s_items = [i for i in main if i["condition"] == "S"]
    s_items = sorted(s_items, key=lambda i: i["id"])[: cfg["packed"]["n_items"]]
    result["packed_vs_separate"] = packed_vs_separate(client, s_items)
    result["consistency"] = consistency(client, cfg["consistency"],
                                        tc.load(cfg["dataset"], split, ["S", "I-hard"],
                                                limit=cfg["consistency"]["n_items"], seed=seed))

    chars = sum(r["state_chars"] * cfg["latency"]["reps"] for r in result["latency"])
    toks = sum(r["input_tokens_mean"] * cfg["latency"]["reps"] for r in result["latency"])
    result["tokens_per_1k_chars"] = 1000 * toks / chars if chars else None
    run.summary = {"tokens_per_1k_chars": result["tokens_per_1k_chars"],
                   "repeat_flip_rate_max": result["consistency"]["repeat_flip_rate_max"]}
    run.write("E0.json", result)
    return result
