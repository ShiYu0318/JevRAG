"""Calibration metrics. Binary inputs are P(label=1); multiclass inputs are
lists of per-class probability dicts."""
from __future__ import annotations

import math
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


# ---- temperature scaling ---------------------------------------------------
_EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def apply_temperature(probs: Sequence[float], t: float) -> List[float]:
    return [1 / (1 + math.exp(-logit(p) / t)) for p in probs]


def apply_temperature_multiclass(probs: Sequence[Mapping[str, float]], t: float) -> List[Dict[str, float]]:
    out = []
    for p in probs:
        logs = {k: math.log(max(v, _EPS)) / t for k, v in p.items()}
        m = max(logs.values())
        z = sum(math.exp(v - m) for v in logs.values())
        out.append({k: math.exp(v - m) / z for k, v in logs.items()})
    return out


def _nll(probs: Sequence[float], labels: Sequence[int]) -> float:
    return -sum(math.log(max(p if y else 1 - p, _EPS)) for p, y in zip(probs, labels)) / max(len(probs), 1)


def fit_temperature(probs: Sequence[float], labels: Sequence[int], lo: float = 0.05, hi: float = 20.0,
                    iters: int = 80) -> float:
    """Temperature minimising binary NLL, by golden-section search on log T."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    f = lambda x: _nll(apply_temperature(probs, math.exp(x)), labels)  # noqa: E731
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return math.exp((a + b) / 2)


def fit_platt(scores: Sequence[float], labels: Sequence[int], iters: int = 100) -> Tuple[float, float]:
    """Platt scaling: fit P(y=1) = sigmoid(a * score + b).

    Uses Platt's smoothed targets so (nearly) separable data still has a finite
    optimum, and damped Newton steps with backtracking so every step lowers the loss.
    """
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    hi, lo = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
    targets = [hi if y else lo for y in labels]

    def loss(a: float, b: float) -> float:
        total = 0.0
        for s, t in zip(scores, targets):
            z = a * s + b
            # log(1 + e^z) computed stably
            softplus = z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))
            total += softplus - t * z
        return total

    a, b = 0.0, math.log((n_pos + 1) / (n_neg + 1))
    cur = loss(a, b)
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for s, t in zip(scores, targets):
            z = a * s + b
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            w = max(p * (1 - p), 1e-12)
            ga += (p - t) * s
            gb += p - t
            haa += w * s * s
            hab += w * s
            hbb += w
        haa += 1e-12
        hbb += 1e-12
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-10:
            na, nb = a - step * da, b - step * db
            new = loss(na, nb)
            if new < cur:
                break
            step /= 2
        else:
            break
        if cur - new < 1e-10 * max(1.0, abs(cur)):
            a, b, cur = na, nb, new
            break
        a, b, cur = na, nb, new
    return a, b


def apply_platt(scores: Sequence[float], a: float, b: float) -> List[float]:
    return [1 / (1 + math.exp(-max(-35.0, min(35.0, a * s + b)))) for s in scores]
