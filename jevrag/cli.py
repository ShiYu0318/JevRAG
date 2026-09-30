"""Command line entry point.

    python -m jevrag build --splits dev,test
    python -m jevrag run E0 --backend jev --split dev
    python -m jevrag tables
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

from . import __version__
from .env import load_dotenv


def _retriever(paras, depth: int, dense: bool, cache_dir: Path):
    from .retrieval import BM25, rrf

    ids = sorted(paras)
    t0 = time.time()
    bm25 = BM25(ids, [paras[i].text for i in ids], unigrams=False)
    print(f"bm25 index: {len(ids)} passages in {time.time() - t0:.1f}s", file=sys.stderr)
    index = None
    if dense:
        from .retrieval.dense import DenseIndex
        index = DenseIndex(ids, [paras[i].text for i in ids], cache=cache_dir / "bge-m3.npy")

    def retrieve(queries: List[str]) -> List[List[Tuple[str, float]]]:
        lex = [bm25.search(q, depth) for q in queries]
        if index is None:
            return lex
        den = index.search_many(queries, depth)
        return [rrf([x, y], top=depth) for x, y in zip(lex, den)]
    return retrieve


def cmd_build(a: argparse.Namespace) -> None:
    try:
        from .data import build as B
    except ImportError:
        sys.exit("the dataset builder is not included in this release yet")
    from .data import drcd

    if a.download:
        drcd.download(a.raw)
    splits = a.splits.split(",")
    paras, questions = drcd.load(a.raw)  # the pool always holds every split
    retrieve = _retriever(paras, a.depth, a.dense, Path(a.out).parent)

    cfg = B.BuildConfig(k=a.k, seed=a.seed, pool_depth=a.depth,
                        conditions=tuple(a.conditions.split(",")) if a.conditions else B.BuildConfig.conditions)
    builder = B.Builder(paras, retrieve, cfg)
    out = Path(a.out)
    meta = {"version": B.VERSION, "k": a.k, "seed": a.seed, "retrieval": "bm25+bge-m3 rrf" if a.dense else "bm25",
            "opencc": B.opencc_available(), "splits": {}}
    for s in splits:
        t0 = time.time()
        rows = builder.build_split(s, questions[s])
        simp = [B.to_simplified(r) for r in rows] if B.opencc_available() and not a.no_simplified else []
        meta["splits"][s] = {"sha256": B.write_jsonl(out / f"{s}.jsonl", rows), **B.stats(rows)}
        if simp:
            meta["splits"][s]["zh-Hans_sha256"] = B.write_jsonl(out / f"{s}.zh-Hans.jsonl", simp)
        print(f"{s}: {len(rows)} instances in {time.time() - t0:.1f}s {meta['splits'][s]['by_condition']}",
              file=sys.stderr)
    (out / "build.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_pools(a: argparse.Namespace) -> None:
    from .data import drcd
    from .data.jsonl import write_rows
    from .data.pools import build_pools

    paras, questions = drcd.load(a.raw)
    retrieve = _retriever(paras, max(a.depth, 20), a.dense, Path(a.out).parent)
    out = Path(a.out)
    meta = {"depth": a.depth, "retrieval": "bm25+bge-m3 rrf" if a.dense else "bm25", "splits": {}}
    for s in a.splits.split(","):
        rows = build_pools(questions[s], retrieve, a.depth)
        hit = sum(1 for r in rows if any(c["pid"] in r["qrels"] for c in r["candidates"])) / max(len(rows), 1)
        meta["splits"][s] = {"sha256": write_rows(out / f"{s}.jsonl", rows), "n": len(rows),
                             f"gold_recall@{a.depth}": round(hit, 4)}
        print(f"{s}: {len(rows)} pools, gold recall@{a.depth} {hit:.3f}", file=sys.stderr)
    (out / "pools.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_audit(a: argparse.Namespace) -> None:
    try:
        from .data import audit
    except ImportError:
        sys.exit("the label audit is not included in this release yet")
    from .data import tc

    if a.score:
        print(json.dumps(audit.score_sheet(a.score), ensure_ascii=False, indent=2))
        return
    rows = tc.load(a.dataset, a.split, a.conditions.split(",") if a.conditions else None)
    sample = audit.sample_for_audit(rows, a.n, a.seed)
    flags = None
    if not a.no_nli:
        from .baselines.nli import NLIScorer
        flags = audit.flag(sample, NLIScorer(), a.tau)
        summary = audit.flag_summary(sample, flags)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).with_suffix(".flags.json").write_text(
            json.dumps({"summary": summary, "tau": a.tau, "flags": flags}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    audit.write_sheet(a.out, sample, flags)
    print(a.out)


def cmd_run(a: argparse.Namespace) -> None:
    from .backends import get_backend
    from .cache import CallCache
    from .config import load_config
    from .experiments import EXPERIMENTS
    from .runlog import Run

    exp = a.experiment.upper()
    if exp not in EXPERIMENTS:
        sys.exit(f"unknown experiment {exp}; available: {', '.join(EXPERIMENTS)}")
    cfg = load_config(a.config) if a.config else {}
    budget = cfg.get("budget_usd")
    client = get_backend(a.backend, cache=None if a.no_cache else CallCache(a.cache),
                         pin_version=a.pin_version, budget_usd=budget)
    datasets = {}
    for key in ("dataset", "length_dataset"):
        if key in cfg:
            meta = Path(cfg[key]) / "build.json"
            if meta.exists():
                datasets[key] = json.loads(meta.read_text(encoding="utf-8"))["splits"].get(a.split, {}).get("sha256")
    with Run(exp, client.name, a.split, cfg, root=a.results, datasets=datasets) as run:
        run.attach(client)
        EXPERIMENTS[exp].run(client, run, a.split, cfg)
        print(run.dir)


def cmd_demo(a: argparse.Namespace) -> None:
    from .demo import serve

    serve(a.port, a.results, a.dataset, a.milestones, None if a.no_cache else a.cache)


def cmd_tables(a: argparse.Namespace) -> None:
    from .tables import make_tables

    for p in make_tables(a.results, a.out):
        print(p)


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevrag")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the benchmark from DRCD")
    b.add_argument("--raw", default="data/raw")
    b.add_argument("--out", default="data/build/jevrag-tc-v0.1.0-k3")
    b.add_argument("--splits", default="dev,test")
    b.add_argument("--k", type=int, default=3)
    b.add_argument("--seed", type=int, default=20261005)
    b.add_argument("--depth", type=int, default=50)
    b.add_argument("--conditions", default="")
    b.add_argument("--dense", action="store_true", help="add bge-m3 and fuse with RRF")
    b.add_argument("--download", action="store_true")
    b.add_argument("--no-simplified", action="store_true")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="run one experiment")
    r.add_argument("experiment")
    r.add_argument("--backend", required=True)
    r.add_argument("--split", default="dev")
    r.add_argument("--config")
    r.add_argument("--results", default="results")
    r.add_argument("--cache", default="cache/calls.sqlite")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--pin-version")
    r.set_defaults(fn=cmd_run)

    pl = sub.add_parser("pools", help="freeze rerank candidate pools")
    pl.add_argument("--raw", default="data/raw")
    pl.add_argument("--out", default="data/build/pools-v0.1.0")
    pl.add_argument("--splits", default="dev,test")
    pl.add_argument("--depth", type=int, default=20)
    pl.add_argument("--dense", action="store_true")
    pl.set_defaults(fn=cmd_pools)

    au = sub.add_parser("audit", help="flag likely label errors and write audit sheets")
    au.add_argument("--dataset", default="data/build/jevrag-tc-v0.1.0-k3")
    au.add_argument("--split", default="dev")
    au.add_argument("--conditions", default="I-hard,I-easy,P")
    au.add_argument("--n", type=int, default=200)
    au.add_argument("--seed", type=int, default=20261005)
    au.add_argument("--tau", type=float, default=0.5)
    au.add_argument("--out", default="docs/audit/dev_sheet.csv")
    au.add_argument("--no-nli", action="store_true")
    au.add_argument("--score", help="score a filled sheet instead of writing one")
    au.set_defaults(fn=cmd_audit)

    dm = sub.add_parser("demo", help="local console: results, milestones and a live judge")
    dm.add_argument("--port", type=int, default=8900)
    dm.add_argument("--results", default="results")
    dm.add_argument("--dataset", default="data/build/jevrag-tc-v0.1.0-k3")
    dm.add_argument("--milestones", default="docs/milestones.json")
    dm.add_argument("--cache", default="cache/calls.sqlite")
    dm.add_argument("--no-cache", action="store_true")
    dm.set_defaults(fn=cmd_demo)

    t = sub.add_parser("tables", help="render reports from results/")
    t.add_argument("--results", default="results")
    t.add_argument("--out", default="docs/results")
    t.set_defaults(fn=cmd_tables)

    a = ap.parse_args(argv)
    load_dotenv()
    a.fn(a)


if __name__ == "__main__":
    main()
