from .ranking import mrr_at_k, ndcg_at_k, recall_at_k


def percentile(xs, q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))]
