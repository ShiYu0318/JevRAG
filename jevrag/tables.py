"""Render markdown tables from results/ only. Never edit the output by hand."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _latest(results: Path, exp: str) -> Dict[str, Path]:
    """Most recent finished run directory per backend for one experiment.

    A run that stopped early (budget, HTTP error) has a manifest but no result file; skip it.
    """
    runs: Dict[str, Path] = {}
    for m in sorted(results.glob(f"{exp}-*/manifest.json")):
        if not (m.parent / f"{exp}.json").exists():
            continue
        man = json.loads(m.read_text(encoding="utf-8"))
        runs[f"{man['backend']} / {man['split']}"] = m.parent
    return runs


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) < 10 else f"{x:.0f}"
    return str(x)


def _table(header: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def e0_report(results: Path) -> str:
    parts = ["# E0 backend profile", ""]
    for backend, d in sorted(_latest(results, "E0").items()):
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        res = json.loads((d / "E0.json").read_text(encoding="utf-8"))
        parts += [f"## {backend}", "",
                  f"run `{man['run_id']}` · versions `{json.dumps(man['model_versions'], ensure_ascii=False)}`"
                  f" · commit `{man['git_commit'][:10]}`", ""]
        parts += ["### Latency", "", _table(
            ["state chars", "questions", "p50 ms", "p95 ms", "input tokens"],
            [[r["state_chars"], r["n_questions"], r["p50_ms"], r["p95_ms"], r["input_tokens_mean"]]
             for r in res["latency"]]), ""]
        if isinstance(res["length"], list):
            parts += ["### Accuracy vs passages", "", _table(
                ["passages", "n", "accuracy", "median chars"],
                [[r["n_passages"], r["n"], r["accuracy"], r["state_chars_median"]] for r in res["length"]]), ""]
        else:
            parts += [f"Length curve: {res['length']}", ""]
        pv = res["packed_vs_separate"]
        parts += ["### Pointwise vs packed", "", _table(
            ["mode", "gold top-1", "p50 ms", "p95 ms"],
            [[m, pv[m]["gold_top1"], pv[m]["p50_ms"], pv[m]["p95_ms"]] for m in ("pointwise", "packed")]),
            "", f"Answer-label agreement: {_fmt(pv['answer_label_agreement'])}", ""]
        c = res["consistency"]
        parts += ["### Consistency", "",
                  f"Repeat flip rate (max over {len(c['repeat_flip_rates'])}): {_fmt(c['repeat_flip_rate_max'])}",
                  "", "Paraphrase flip rates: " + ", ".join(f"{k} {_fmt(v)}" for k, v in
                                                            c["paraphrase_flip_rates"].items()),
                  "", f"Tokens per 1k chars: {_fmt(res['tokens_per_1k_chars'])}", ""]
    return "\n".join(parts)


def _load(d: Path, exp: str):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    res = json.loads((d / f"{exp}.json").read_text(encoding="utf-8"))
    head = [f"run `{man['run_id']}` · versions `{json.dumps(man['model_versions'], ensure_ascii=False)}`"
            f" · commit `{man['git_commit'][:10]}` · cost ${man['cost_usd']:.4f}", ""]
    return man, res, head


def _ci(x) -> str:
    return f"{x['value']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"


def e1_report(results: Path) -> str:
    parts = ["# E1 rerank", ""]
    for key, d in sorted(_latest(results, "E1").items()):
        man, res, head = _load(d, "E1")
        ref = man["config"].get("reference", "rrf")
        parts += [f"## {key}", ""] + head
        parts += [f"{res['n_items']} questions, gold in pool {_fmt(res['pool_gold_recall@20'])}", ""]
        rows = []
        for m, r in res["methods"].items():
            vs = r.get("vs_reference_ndcg@10")
            rows.append([m, _ci(r["ndcg@10"]), _ci(r["mrr@10"]), _ci(r["recall@5"]),
                         f"{vs['diff']:+.3f} [{vs['lo']:+.3f}, {vs['hi']:+.3f}]" if vs else "",
                         ("yes" if vs["non_inferior"] else "no") if vs else ""])
        parts += [_table(["method", "nDCG@10", "MRR@10", "Recall@5", f"Δ nDCG vs {ref}", "non-inferior"], rows), ""]
        if res.get("packed_subset"):
            parts += ["Packed subset:", "", _table(["method", "nDCG@10"], [[m, _ci(r["ndcg@10"])] for m, r in
                                                                         res["packed_subset"].items()]), ""]
        lat = res["latency_ms_per_question"]
        cached = man["calls"] and man["cache_hits"] == man["calls"]
        latency = ("Latency not measured: every call was served from the cache" if cached else
                   f"Latency per question (20 passages): p50 {_fmt(lat['p50'])} ms, p95 {_fmt(lat['p95'])} ms")
        parts += [f"{latency} · cost per 1k questions ${res['cost_per_1k_questions']:.4f}", ""]
    return "\n".join(parts)


def e2_report(results: Path) -> str:
    parts = ["# E2 calibration", ""]
    cols = ["ece_width", "ece_mass", "brier", "auroc", "aurc", "confident_error_rate@0.9", "coverage@risk0.05"]
    for key, d in sorted(_latest(results, "E2").items()):
        _, res, head = _load(d, "E2")
        parts += [f"## {key}", ""] + head
        for label in ("raw", "scaled", "platt_scaled"):
            if label in res:
                parts += [f"### {label}", "", _table(["group", "n", "pos rate"] + cols,
                          [[g, r["n"], r["positive_rate"]] + [r[c] for c in cols] for g, r in res[label].items()]), ""]
        for name, entry in res.get("baselines", {}).items():
            for label in ("raw_sigmoid", "platt_scaled"):
                if label in entry:
                    parts += [f"### {name} ({label})", "", _table(["group", "n", "pos rate"] + cols,
                              [[g, r["n"], r["positive_rate"]] + [r[c] for c in cols]
                               for g, r in entry[label].items()]), ""]
        if "temperature" in res:
            bound = [k for k, v in res.get("temperature_at_search_bound", {}).items() if v]
            parts += [f"Temperature (fitted on the fit split): passage {_fmt(res['temperature']['passage'])}, "
                      f"set {_fmt(res['temperature']['set'])}"
                      + (f" (at the search bound for {', '.join(bound)})" if bound else ""), ""]
        if "platt" in res:
            parts += ["Platt on logit: " + ", ".join(f"{k} a={_fmt(v['a'])} b={_fmt(v['b'])}"
                                                    for k, v in res["platt"].items()), ""]
        h = res["h2"]
        parts += [f"H2: ECE easy {_fmt(h['ece_easy'])}, hard {_fmt(h['ece_hard'])}, gap {_fmt(h['gap'])}"
                  f" → {'supported' if h['supported'] else 'not supported'}", ""]
    return "\n".join(parts)


def e3_report(results: Path) -> str:
    parts = ["# E3 evidence sufficiency", ""]
    classes = ["sufficient", "partial", "conflicting", "insufficient"]
    for key, d in sorted(_latest(results, "E3").items()):
        _, res, head = _load(d, "E3")
        parts += [f"## {key}", ""] + head + [f"n = {res['n']} {res['n_by_condition']}", ""]
        rows = [["set"] + [_ci(res["set"]["macro_f1"])] + [res["set"]["per_class_f1"][c] for c in classes]]
        for a, r in res["per_passage"].items():
            rows.append([f"per-passage {a}", _ci(r["macro_f1"])] + [r["per_class_f1"][c] for c in classes])
        for a, r in res["two_stage"].items():
            rows.append([f"two-stage {a}", _ci(r["macro_f1"])] + [r["per_class_f1"][c] for c in classes])
        rows.append(["majority", _ci(res["majority"]["macro_f1"])] + [res["majority"]["per_class_f1"][c]
                                                                     for c in classes])
        if "nli" in res:
            rows.append(["NLI (gold answer, optimistic)", _ci(res["nli"]["macro_f1"])]
                        + [res["nli"]["per_class_f1"][c] for c in classes])
        parts += [_table(["method", "macro-F1"] + [f"F1 {c}" for c in classes], rows), ""]
        acc = res["set"]["accuracy_by_condition"]
        parts += ["Set-level accuracy by condition: " + ", ".join(f"{c} {_fmt(v)}" for c, v in acc.items()), "",
                  f"Set-level top-label ECE {_fmt(res['set']['top_label_ece'])}", "",
                  f"Conflicting predicted as: {res['conflict_predicted_as']}", "",
                  "Evidence AUROC: " + ", ".join(f"{a} {_fmt(r['evidence_auroc'])}"
                                                 for a, r in res["per_passage"].items()), "",
                  f"Thresholds: {json.dumps(res['thresholds'])}", "",
                  f"Extra conditions: {json.dumps(res['extra'])}", ""]
    return "\n".join(parts)


def e4_report(results: Path) -> str:
    parts = ["# E4 script and instruction language", ""]
    for key, d in sorted(_latest(results, "E4").items()):
        _, res, head = _load(d, "E4")
        parts += [f"## {key}", ""] + head + [f"n = {res['n']}", ""]
        rows = []
        for k, r in res["settings"].items():
            fl = r.get("flip_rate_vs_base")
            rows.append([k, r["macro_f1"], r["accuracy"], r["top_label_ece"], r["mean_confidence"],
                         _ci(fl) if fl else "base", _fmt(r.get("passage_flip_rate_vs_base", ""))])
        parts += [_table(["script / instructions", "macro-F1", "accuracy", "ECE", "mean conf",
                          "set flip vs base", "passage flip"], rows), ""]
    return "\n".join(parts)


REPORTS = {"E0": ("E0_backend_profile.md", e0_report), "E1": ("E1_rerank.md", e1_report),
           "E2": ("E2_calibration.md", e2_report), "E3": ("E3_sufficiency.md", e3_report),
           "E4": ("E4_script_robustness.md", e4_report)}


def make_tables(results: str | Path = "results", out: str | Path = "docs/results") -> List[Path]:
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for exp, (name, fn) in REPORTS.items():
        if _latest(results, exp):
            p = out / name
            p.write_text(fn(results) + "\n", encoding="utf-8")
            written.append(p)
    return written
