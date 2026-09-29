"""Run every RAG decision point on the toy set, for one or more backends.

    python scripts/run_toy.py --backends jev,kev4b,laya-ml
    python scripts/run_toy.py --backends mock          # plumbing test

Writes results/toy_<backend>.json and prints a comparison table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevrag import ops  # noqa: E402
from jevrag.backends import SystemOneError, get_backend  # noqa: E402
from jevrag.metrics import mrr_at_k, ndcg_at_k  # noqa: E402

VARIANT_TO_VERDICT = {"sufficient": "sufficient", "insufficient": "insufficient",
                      "conflicting": "conflicting", "injected": "sufficient",
                      "false_premise": "insufficient"}


def run(backend: str) -> dict:
    client = get_backend(backend)
    data = json.loads((ROOT / "data" / "toy_zh.json").read_text(encoding="utf-8"))
    corpus, items = data["corpus"], data["items"]
    lat, res = [], {"backend": client.name}

    # 1) routing
    hits = 0
    for it in items:
        r = ops.route_query(client, it["question"], data["routes"])
        lat.append(r.latency_ms)
        hits += r["route"].label == data["route_labels"][it["id"]]
    res["routing_acc"] = hits / len(items)

    # 2) retrieval policy (Adaptive-RAG / Self-RAG retrieve token)
    res["policy"] = {}
    for it in items:
        r = ops.retrieval_policy(client, it["question"])
        lat.append(r.latency_ms)
        res["policy"][it["id"]] = {"complexity": r["complexity"].probs, "needs_fresh": r["needs_fresh"].p}

    # 3) reranking the whole corpus, pointwise vs packed
    ids = list(corpus)
    for mode in ("pointwise", "packed"):
        nd, mr = [], []
        for it in items:
            g = ops.grade_passages(client, it["question"], [corpus[i] for i in ids], mode=mode,
                                   with_safety=(mode == "pointwise"))
            ranked = [ids[x.index] for x in sorted(g, key=lambda x: (x.answers_p, x.relevance), reverse=True)]
            nd.append(ndcg_at_k(ranked, it["qrels"], 5))
            mr.append(mrr_at_k(ranked, it["qrels"], 5))
        res[f"ndcg@5_{mode}"] = statistics.mean(nd)
        res[f"mrr@5_{mode}"] = statistics.mean(mr)

    # 4) injection flag on the planted passage
    g = ops.grade_passages(client, items[0]["question"], [corpus["d7"], corpus["d1"]])
    res["injection_p_planted_vs_clean"] = [round(g[0].injection_p, 3), round(g[1].injection_p, 3)]

    # 5) sufficiency verdicts per counterfactual variant
    rows, correct = [], 0
    for it in items:
        for variant, docs in it["variants"].items():
            r = ops.assess_sufficiency(client, it["question"], [corpus[d] for d in docs])
            lat.append(r.latency_ms)
            want = VARIANT_TO_VERDICT[variant]
            got = r["verdict"].label
            correct += got == want
            rows.append({"q": it["id"], "variant": variant, "want": want, "got": got,
                         "action": ops.decide_action(r), "probs": r["verdict"].probs})
    res["sufficiency_acc"] = correct / len(rows)
    res["sufficiency_rows"] = rows

    # 6) claim checks: one supported, one hallucinated sentence
    checks = ops.check_claims(client,
                              ["中央大學於 1962 年在臺灣復校。", "中央大學於 1962 年在臺北市復校。"],
                              [corpus["d1"]])
    res["claim_support_p"] = [round(c.support_p, 3) for c in checks]

    res["latency_ms_median"] = statistics.median(lat)
    res["latency_ms_p95"] = sorted(lat)[int(0.95 * (len(lat) - 1))]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="mock")
    a = ap.parse_args()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    table = []
    for b in a.backends.split(","):
        try:
            r = run(b.strip())
        except SystemOneError as e:
            print(f"[skip] {b}: {e}")
            continue
        (out_dir / f"toy_{b}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        table.append(r)
    cols = ["routing_acc", "ndcg@5_pointwise", "ndcg@5_packed", "sufficiency_acc",
            "claim_support_p", "injection_p_planted_vs_clean", "latency_ms_median", "latency_ms_p95"]
    print("| backend | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for r in table:
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        print(f"| {r['backend']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
