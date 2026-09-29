"""E1: passage relevance and reranking on frozen candidate pools."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from ..backends.systemone import SystemOneClient
from ..data import drcd
from ..data.pools import load_pools, texts
from ..metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from ..ops.grade import PassageGrade, grade_passages, rank
from ..runlog import Run
from ..stats import non_inferior, paired_bootstrap_diff
from .common import ci, cost_per_1k, merge, passage_grades, sample

DEFAULTS: Dict[str, Any] = {
    "pools": "data/build/pools-v0.1.0",
    "raw": "data/raw",
    "n_items": 300,
    "packed_n_items": 50,
    "keys": ["noul", "score", "noul_then_score"],
    "reference": "rrf",
    "margin": 0.01,  # non-inferiority margin on nDCG@10 (1 point)
    "baselines": [],
    "workers": 8,
    "seed": 7,
    "bootstrap": 2000,
}


def _metrics(ranked: List[str], qrels: Dict[str, int]) -> Dict[str, float]:
    return {"ndcg@10": ndcg_at_k(ranked, qrels, 10), "mrr@10": mrr_at_k(ranked, qrels, 10),
            "recall@5": recall_at_k(ranked, qrels, 5)}


def _summarise(per_q: Dict[str, List[Dict[str, float]]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ref = per_q.get(cfg["reference"])
    for method, rows in per_q.items():
        n = len(rows)
        res = {m: ci(n, lambda idx, m=m, rows=rows: sum(rows[i][m] for i in idx) / len(idx), n=cfg["bootstrap"])
               for m in ("ndcg@10", "mrr@10", "recall@5")}
        if ref is not None and method != cfg["reference"] and len(ref) == n:
            d = paired_bootstrap_diff(n, lambda idx: sum(rows[i]["ndcg@10"] for i in idx) / len(idx),
                                      lambda idx: sum(ref[i]["ndcg@10"] for i in idx) / len(idx),
                                      n=cfg["bootstrap"])
            d["non_inferior"] = non_inferior(d["lo"], cfg["margin"])
            res["vs_reference_ndcg@10"] = d
        out[method] = res
    return out


def run(client: SystemOneClient, run: Run, split: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = merge(DEFAULTS, overrides)
    paras, _ = drcd.load(cfg["raw"])
    pools = sample(load_pools(f"{cfg['pools']}/{split}.jsonl"), cfg["n_items"], cfg["seed"], key="qid")
    per_q: Dict[str, List[Dict[str, float]]] = {"rrf": []}
    per_q.update({f"{client.name}:pointwise:{k}": [] for k in cfg["keys"]})
    ms: List[float] = []
    cand_log = (run.dir / "E1_candidates.jsonl").open("w", encoding="utf-8")

    for pool in pools:
        pids = [c["pid"] for c in pool["candidates"]]
        per_q["rrf"].append(_metrics(pids, pool["qrels"]))
        t0 = time.perf_counter()
        resp = passage_grades(client, pool["question"], texts(pool, paras), workers=cfg["workers"],
                              tag={"part": "rerank", "qid": pool["qid"]})
        ms.append((time.perf_counter() - t0) * 1000)
        grades = [PassageGrade(i, r["rel"].expected_score, r["ans"].p) for i, r in enumerate(resp)]
        for k in cfg["keys"]:
            per_q[f"{client.name}:pointwise:{k}"].append(_metrics([pids[i] for i in rank(grades, k)], pool["qrels"]))
        for g, pid in zip(grades, pids):
            cand_log.write(json.dumps({"qid": pool["qid"], "pid": pid, "gold": pid in pool["qrels"],
                                       "p_ans": g.answers_p, "rel": g.relevance}) + "\n")
    cand_log.close()

    packed_n = min(cfg["packed_n_items"], len(pools))
    if packed_n:
        key = f"{client.name}:packed:noul_then_score"
        per_q[key] = []
        sub = pools[:packed_n]
        for pool in sub:
            g = grade_passages(client, pool["question"], texts(pool, paras), mode="packed")
            per_q[key].append(_metrics([pool["candidates"][i]["pid"] for i in rank(g)], pool["qrels"]))
        # pointwise on the same subset, so the two modes are paired
        pw = f"{client.name}:pointwise:noul_then_score"
        if pw in per_q:
            per_q[pw + "@packed_subset"] = per_q[pw][:packed_n]

    for b in cfg["baselines"]:
        per_q[b] = _baseline(b, pools, paras)

    result = {
        "n_items": len(pools),
        "pool_gold_recall@20": sum(1 for p in pools if any(c["pid"] in p["qrels"] for c in p["candidates"])) / len(pools),
        "methods": _summarise({k: v for k, v in per_q.items() if len(v) == len(pools)}, cfg),
        "packed_subset": _summarise({k: v for k, v in per_q.items() if len(v) == packed_n and packed_n
                                     and ("packed" in k)}, {**cfg, "reference": ""}) if packed_n else {},
        "latency_ms_per_question": {"p50": percentile(ms, 50), "p95": percentile(ms, 95)},
        "cost_per_1k_questions": cost_per_1k(client, len(pools)),
    }
    run.summary = {m: r["ndcg@10"]["value"] for m, r in result["methods"].items()}
    run.write("E1.json", result)
    return result


def _baseline(name: str, pools: List[dict], paras) -> List[Dict[str, float]]:
    if name == "bge-reranker":
        from ..baselines.cross_encoder import CrossEncoderReranker
        model = CrossEncoderReranker()
        out = []
        for pool in pools:
            scores = model.score(pool["question"], texts(pool, paras))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            out.append(_metrics([pool["candidates"][i]["pid"] for i in order], pool["qrels"]))
        return out
    raise ValueError(f"unknown baseline {name!r}")
