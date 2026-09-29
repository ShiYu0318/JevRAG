from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

Ranked = Sequence[Tuple[str, float]]


def rrf(runs: Sequence[Ranked], k: int = 60, top: int = 50) -> List[Tuple[str, float]]:
    """Reciprocal rank fusion of several ranked lists."""
    acc: Dict[str, float] = defaultdict(float)
    for run in runs:
        for rank, (doc, _) in enumerate(run, 1):
            acc[doc] += 1.0 / (k + rank)
    return sorted(acc.items(), key=lambda x: -x[1])[:top]


def weighted(lexical: Ranked, dense: Ranked, alpha: float, top: int = 50) -> List[Tuple[str, float]]:
    """Min-max normalised linear fusion; alpha is the weight on dense."""
    def norm(run: Ranked) -> Dict[str, float]:
        if not run:
            return {}
        vals = [s for _, s in run]
        lo, hi = min(vals), max(vals)
        return {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s in run}
    a, b = norm(lexical), norm(dense)
    acc = {d: (1 - alpha) * a.get(d, 0.0) + alpha * b.get(d, 0.0) for d in set(a) | set(b)}
    return sorted(acc.items(), key=lambda x: -x[1])[:top]
