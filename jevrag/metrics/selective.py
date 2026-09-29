"""Selective prediction: risk-coverage, AURC, coverage at a risk budget."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def risk_coverage(conf: Sequence[float], correct: Sequence[int]) -> Tuple[List[float], List[float]]:
    """Answer the top-c fraction by confidence; risk = error rate among answered."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    covs, risks, errs = [], [], 0
    for n, i in enumerate(order, 1):
        errs += 1 - correct[i]
        covs.append(n / len(order))
        risks.append(errs / n)
    return covs, risks


def aurc(conf: Sequence[float], correct: Sequence[int]) -> float:
    _, risks = risk_coverage(conf, correct)
    return sum(risks) / len(risks) if risks else 0.0


def risk_at_coverage(conf: Sequence[float], correct: Sequence[int], coverage: float = 0.8) -> float:
    covs, risks = risk_coverage(conf, correct)
    for c, r in zip(covs, risks):
        if c >= coverage - 1e-12:
            return r
    return risks[-1] if risks else 0.0


def coverage_at_risk(conf: Sequence[float], correct: Sequence[int], risk: float = 0.05) -> float:
    """Largest coverage whose risk stays <= ``risk``."""
    covs, risks = risk_coverage(conf, correct)
    best = 0.0
    for c, r in zip(covs, risks):
        if r <= risk + 1e-12:
            best = c
    return best
