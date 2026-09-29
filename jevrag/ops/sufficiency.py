"""Evidence sufficiency: set-level verdicts, per-passage aggregation, actions."""
from __future__ import annotations

import math
from typing import Dict, Sequence

from ..backends.systemone import Response, SystemOneClient
from ..questions import sufficiency_v1


def assess_sufficiency(client: SystemOneClient, query: str, passages: Sequence[str],
                       lang: str = "en") -> Response:
    return client.ask(*sufficiency_v1(query, passages, lang=lang))


def aggregate(ps: Sequence[float], how: str = "noisy_or") -> float:
    """Collapse per-passage P(contains answer) into one evidence score."""
    if not ps:
        return 0.0
    if how == "max":
        return max(ps)
    if how == "noisy_or":
        return 1.0 - math.prod(1.0 - p for p in ps)
    if how == "sum":
        return sum(ps)
    raise ValueError(f"unknown aggregation {how!r}")


def decide_action(verdict: Response, tau_answer: float = 0.6, tau_premise: float = 0.7,
                  tau_conflict: float = 0.4) -> str:
    """Plain-code policy on top of the verdict probabilities."""
    probs: Dict[str, float] = verdict["verdict"].probs
    if verdict["false_premise"].p >= tau_premise:
        return "correct_premise"
    if probs.get("sufficient", 0) >= tau_answer:
        return "answer"
    if probs.get("conflicting", 0) >= tau_conflict:
        return "answer_with_both_sides"
    if probs.get("partial", 0) + probs.get("sufficient", 0) >= tau_answer:
        return "answer_partial_and_flag_gap"
    return "abstain_or_search_web"
