from __future__ import annotations

import math
from typing import Dict, List, Sequence


def _dcg(rels: List[int]) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Dict[str, int], k: int = 10) -> float:
    ideal = _dcg(sorted(qrels.values(), reverse=True)[:k])
    return _dcg([qrels.get(d, 0) for d in ranked_ids[:k]]) / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_ids: Sequence[str], qrels: Dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked_ids[:k]):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], qrels: Dict[str, int], k: int = 10) -> float:
    rel = {d for d, r in qrels.items() if r > 0}
    if not rel:
        return 0.0
    return len(rel & set(ranked_ids[:k])) / len(rel)
