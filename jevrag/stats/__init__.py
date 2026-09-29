"""Paired bootstrap, McNemar and Holm. Items are the resampling unit."""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

Metric = Callable[[Sequence[int]], float]


def bootstrap_ci(n_items: int, metric: Metric, n: int = 10000, seed: int = 7,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """(point, lo, hi). ``metric`` gets a list of item indices."""
    rng = random.Random(seed)
    idx = list(range(n_items))
    point = metric(idx)
    stats = sorted(metric([rng.randrange(n_items) for _ in idx]) for _ in range(n))
    return point, stats[int(alpha / 2 * n)], stats[min(n - 1, int((1 - alpha / 2) * n))]


def paired_bootstrap_diff(n_items: int, metric_a: Metric, metric_b: Metric, n: int = 10000,
                          seed: int = 7, alpha: float = 0.05) -> Dict[str, float]:
    """CI of metric_a - metric_b on the same resampled items, plus a two-sided p."""
    rng = random.Random(seed)
    idx = list(range(n_items))
    point = metric_a(idx) - metric_b(idx)
    diffs = []
    for _ in range(n):
        s = [rng.randrange(n_items) for _ in idx]
        diffs.append(metric_a(s) - metric_b(s))
    diffs.sort()
    below = sum(1 for d in diffs if d <= 0) / n
    above = sum(1 for d in diffs if d >= 0) / n
    return {"diff": point, "lo": diffs[int(alpha / 2 * n)], "hi": diffs[min(n - 1, int((1 - alpha / 2) * n))],
            "p": min(1.0, 2 * min(below, above))}


def non_inferior(ci_lo: float, margin: float) -> bool:
    """True when the lower CI bound of (new - reference) stays above -margin."""
    return ci_lo > -margin


def mcnemar(correct_a: Sequence[int], correct_b: Sequence[int]) -> Dict[str, float]:
    """Exact McNemar test on paired 0/1 outcomes."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Holm-adjusted p-values and reject flags, in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return [(a, a <= alpha) for a in adj]
