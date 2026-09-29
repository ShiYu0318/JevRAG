"""Render markdown tables from results/ only. Never edit the output by hand."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _latest(results: Path, exp: str) -> Dict[str, Path]:
    """Most recent run directory per backend for one experiment."""
    runs: Dict[str, Path] = {}
    for m in sorted(results.glob(f"{exp}-*/manifest.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        runs[man["backend"]] = m.parent
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


REPORTS = {"E0": ("E0_backend_profile.md", e0_report)}


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
