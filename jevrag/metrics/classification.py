from __future__ import annotations

from typing import Dict, Hashable, Optional, Sequence


def confusion(y_true: Sequence[Hashable], y_pred: Sequence[Hashable],
              classes: Optional[Sequence[Hashable]] = None) -> Dict[Hashable, Dict[Hashable, int]]:
    """confusion[true][pred] = count."""
    classes = list(classes) if classes else sorted(set(y_true) | set(y_pred), key=str)
    m = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        m.setdefault(t, {c: 0 for c in classes})
        m[t][p] = m[t].get(p, 0) + 1
    return m


def per_class_f1(y_true: Sequence[Hashable], y_pred: Sequence[Hashable],
                 classes: Optional[Sequence[Hashable]] = None) -> Dict[Hashable, float]:
    classes = list(classes) if classes else sorted(set(y_true) | set(y_pred), key=str)
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        out[c] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return out


def macro_f1(y_true: Sequence[Hashable], y_pred: Sequence[Hashable],
             classes: Optional[Sequence[Hashable]] = None) -> float:
    f = per_class_f1(y_true, y_pred, classes)
    return sum(f.values()) / len(f) if f else 0.0


def accuracy(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def flip_rate(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Share of paired items whose decision differs between two conditions."""
    return sum(1 for x, y in zip(a, b) if x != y) / max(len(a), 1)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
