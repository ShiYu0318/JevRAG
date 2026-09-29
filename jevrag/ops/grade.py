"""Passage grading: rerank, CRAG evaluator, Self-RAG IsRel, adaptive top-k."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..backends.systemone import SystemOneClient
from ..questions import grade_packed_v1, grade_v1


@dataclass
class PassageGrade:
    index: int
    relevance: float          # expected level in [0, 3]
    answers_p: float          # P(passage contains the answer)
    injection_p: float = 0.0
    contradicts_p: float = 0.0


def grade_passages(client: SystemOneClient, query: str, passages: Sequence[str],
                   mode: str = "pointwise", with_safety: bool = True, lang: str = "en",
                   workers: int = 8) -> List[PassageGrade]:
    """mode="pointwise": one request per passage.
    mode="packed":    one request, all passages in the state, one question each.
                      Cheaper prefill, but long states hurt Kev (trained on
                      <=384-token states) and Laya (512/1,024 default context).
    """
    if mode == "pointwise":
        jobs = [grade_v1(query, p, lang=lang, safety=with_safety) for p in passages]
        out = []
        for i, r in enumerate(client.ask_many(jobs, workers)):
            out.append(PassageGrade(i, r["rel"].expected_score, r["ans"].p,
                                    r["inj"].p if with_safety else 0.0,
                                    r["con"].p if with_safety else 0.0))
        return out
    if mode == "packed":
        r = client.ask(*grade_packed_v1(query, passages, lang=lang))
        return [PassageGrade(i, r[f"rel_{i}"].expected_score, r[f"ans_{i}"].p) for i in range(len(passages))]
    raise ValueError("mode must be 'pointwise' or 'packed'")


def rank(grades: Sequence[PassageGrade], key: str = "noul_then_score") -> List[int]:
    """Indices sorted best first. key: noul | score | noul_then_score."""
    if key == "noul":
        sort_key = lambda g: g.answers_p  # noqa: E731
    elif key == "score":
        sort_key = lambda g: g.relevance  # noqa: E731
    else:
        sort_key = lambda g: (g.answers_p, g.relevance)  # noqa: E731
    return [g.index for g in sorted(grades, key=sort_key, reverse=True)]


def adaptive_select(grades: List[PassageGrade], tau: float = 0.5, k_max: int = 8,
                    k_min: int = 0, inj_max: float = 0.5) -> List[PassageGrade]:
    """Adaptive top-k: keep passages whose P(answers) clears tau."""
    safe = [g for g in grades if g.injection_p < inj_max]
    ranked = sorted(safe, key=lambda g: (g.answers_p, g.relevance), reverse=True)
    kept = [g for g in ranked if g.answers_p >= tau][:k_max]
    if len(kept) < k_min:
        kept = ranked[:k_min]
    return kept
