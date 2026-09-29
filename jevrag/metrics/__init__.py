from .calibration import (brier, brier_multiclass, classwise_ece, confident_error_rate, ece, ece_from,
                          reliability, top_label_ece)
from .classification import accuracy, auroc, cohen_kappa, confusion, flip_rate, macro_f1, per_class_f1
from .ranking import mrr_at_k, ndcg_at_k, recall_at_k
from .selective import aurc, coverage_at_risk, risk_at_coverage, risk_coverage


def percentile(xs, q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))]
