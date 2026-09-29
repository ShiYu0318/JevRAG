"""Calibration metrics. Binary inputs are P(label=1); multiclass inputs are
lists of per-class probability dicts."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

Bin = Tuple[float, float, int]  # (mean confidence, accuracy, count)


def _bins_equal_width(conf: Sequence[float], n_bins: int) -> List[List[int]]:
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(conf):
        bins[min(int(c * n_bins), n_bins - 1)].append(i)
    return bins


def _bins_equal_mass(conf: Sequence[float], n_bins: int) -> List[List[int]]:
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    n = len(order)
    return [order[b * n // n_bins:(b + 1) * n // n_bins] for b in range(n_bins)]


def reliability(conf: Sequence[float], correct: Sequence[int], n_bins: int = 15,
                scheme: str = "width") -> List[Bin]:
    """Points of a reliability diagram. ``correct`` is 1 when the event happened."""
    split = _bins_equal_width if scheme == "width" else _bins_equal_mass
    out = []
    for idx in split(conf, n_bins):
        if idx:
            out.append((sum(conf[i] for i in idx) / len(idx),
                        sum(correct[i] for i in idx) / len(idx), len(idx)))
    return out


def ece_from(conf: Sequence[float], correct: Sequence[int], n_bins: int = 15,
             scheme: str = "width") -> float:
    n = len(conf)
    if n == 0:
        return 0.0
    return sum(k / n * abs(a - c) for c, a, k in reliability(conf, correct, n_bins, scheme))


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 15, scheme: str = "width") -> float:
    """Binary ECE on P(label=1) against 0/1 labels."""
    return ece_from(probs, labels, n_bins, scheme)


def top_label_ece(probs: Sequence[Mapping[str, float]], labels: Sequence[str], n_bins: int = 15,
                  scheme: str = "width") -> float:
    conf, correct = [], []
    for p, y in zip(probs, labels):
        pred = max(p, key=p.get)
        conf.append(p[pred])
        correct.append(int(pred == y))
    return ece_from(conf, correct, n_bins, scheme)


def classwise_ece(probs: Sequence[Mapping[str, float]], labels: Sequence[str], classes: Sequence[str],
                  n_bins: int = 15, scheme: str = "width") -> Dict[str, float]:
    """One-vs-rest ECE per class."""
    return {c: ece_from([p.get(c, 0.0) for p in probs], [int(y == c) for y in labels], n_bins, scheme)
            for c in classes}


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / max(len(probs), 1)


def brier_multiclass(probs: Sequence[Mapping[str, float]], labels: Sequence[str],
                     classes: Sequence[str]) -> float:
    total = 0.0
    for p, y in zip(probs, labels):
        total += sum((p.get(c, 0.0) - (1.0 if c == y else 0.0)) ** 2 for c in classes)
    return total / max(len(probs), 1)


def confident_error_rate(conf: Sequence[float], correct: Sequence[int], thr: float = 0.9) -> float:
    """Share of all items answered wrongly with confidence >= thr."""
    bad = sum(1 for c, y in zip(conf, correct) if c >= thr and not y)
    return bad / max(len(conf), 1)
