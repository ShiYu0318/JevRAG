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
