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


def cmd_build(a: argparse.Namespace) -> None:
    try:
        from .data import build as B
    except ImportError:
        sys.exit("the dataset builder is not included in this release yet")
    from .data import drcd
    from .retrieval import BM25, rrf

    if a.download:
        drcd.download(a.raw)
    splits = a.splits.split(",")
    paras, questions = drcd.load(a.raw)  # the pool always holds every split
    ids = sorted(paras)
    t0 = time.time()
    bm25 = BM25(ids, [paras[i].text for i in ids], unigrams=False)
    print(f"bm25 index: {len(ids)} passages in {time.time() - t0:.1f}s", file=sys.stderr)
    dense = None
    if a.dense:
        from .retrieval.dense import DenseIndex
        dense = DenseIndex(ids, [paras[i].text for i in ids], cache=Path(a.out).parent / "bge-m3.npy")

    def retrieve(queries: List[str]) -> List[List[Tuple[str, float]]]:
        lex = [bm25.search(q, a.depth) for q in queries]
        if dense is None:
            return lex
        den = dense.search_many(queries, a.depth)
        return [rrf([x, y], top=a.depth) for x, y in zip(lex, den)]

    cfg = B.BuildConfig(k=a.k, seed=a.seed, pool_depth=a.depth,
                        conditions=tuple(a.conditions.split(",")) if a.conditions else B.BuildConfig.conditions)
    builder = B.Builder(paras, retrieve, cfg)
    out = Path(a.out)
    meta = {"version": B.VERSION, "k": a.k, "seed": a.seed, "retrieval": "bm25+bge-m3 rrf" if dense else "bm25",
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

    t = sub.add_parser("tables", help="render reports from results/")
    t.add_argument("--results", default="results")
    t.add_argument("--out", default="docs/results")
    t.set_defaults(fn=cmd_tables)

    a = ap.parse_args(argv)
    load_dotenv()
    a.fn(a)


if __name__ == "__main__":
    main()
