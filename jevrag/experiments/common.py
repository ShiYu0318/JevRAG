"""Shared pieces for the experiment runners.

All experiments grade a passage with ``grade.v1`` (no safety questions) and
judge a passage set with ``sufficiency.v1``, so identical requests hit the call
cache across E1-E4.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence

from ..backends.systemone import Response, SystemOneClient
from ..questions import grade_v1, sufficiency_v1
from ..stats import bootstrap_ci

VERDICTS = ["sufficient", "partial", "conflicting", "insufficient"]


def merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def sample(rows: Sequence[dict], n: int | None, seed: int, key: str = "id") -> List[dict]:
    rows = list(rows)
    if n is None or n >= len(rows):
        return rows
    return sorted(random.Random(seed).sample(rows, n), key=lambda r: r[key])


def by_condition(rows: Sequence[dict], n_per: int | None, seed: int) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["condition"], []).append(r)
    out: List[dict] = []
    for cond in sorted(groups):
        out += sample(groups[cond], n_per, seed)
    return out


def passage_grades(client: SystemOneClient, question: str, passages: Sequence[str], lang: str = "en",
                   workers: int = 8, tag: Dict[str, Any] | None = None) -> List[Response]:
    jobs = [grade_v1(question, p, lang=lang, safety=False) for p in passages]
    return client.ask_many(jobs, workers, tag=tag)


def verdicts(client: SystemOneClient, items: Sequence[dict], lang: str = "en", workers: int = 8,
             part: str = "verdict") -> List[Response]:
    jobs = [sufficiency_v1(i["question"], [c["text"] for c in i["contexts"]], lang=lang) for i in items]
    out: List[Response] = [None] * len(jobs)  # type: ignore[list-item]

    # ask_many shares one tag; per-item tags need one call each, so run them in a pool here.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(client.ask, *job, tag={"part": part, "instance_id": it["id"], "lang": lang}): n
                for n, (job, it) in enumerate(zip(jobs, items))}
        for f, n in futs.items():
            out[n] = f.result()
    return out


def ci(n_items: int, metric: Callable[[List[int]], float], n: int = 2000, seed: int = 7) -> Dict[str, float]:
    point, lo, hi = bootstrap_ci(n_items, metric, n=n, seed=seed)
    return {"value": point, "lo": lo, "hi": hi}


def cost_per_1k(client: SystemOneClient, n_units: int) -> float:
    return 1000 * client.meter.cost_usd / n_units if n_units else 0.0
