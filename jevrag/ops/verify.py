"""Sentence-level faithfulness / citation checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..backends.systemone import SystemOneClient
from ..questions import claim_v1


@dataclass
class ClaimCheck:
    sentence: str
    support_p: float
    cited: List[int] = field(default_factory=list)


def check_claims(client: SystemOneClient, sentences: Sequence[str], evidence: Sequence[str],
                 citations: Optional[Sequence[Sequence[int]]] = None) -> List[ClaimCheck]:
    jobs, cited_all = [], []
    for i, s in enumerate(sentences):
        cited = list(citations[i]) if citations else list(range(len(evidence)))
        cited_all.append(cited)
        jobs.append(claim_v1(s, [evidence[j] for j in cited]))
    return [ClaimCheck(s, r["sup"].p, c) for s, r, c in zip(sentences, client.ask_many(jobs), cited_all)]


def faithfulness(checks: List[ClaimCheck], agg: str = "mean") -> float:
    ps = [c.support_p for c in checks] or [1.0]
    return min(ps) if agg == "min" else sum(ps) / len(ps)
