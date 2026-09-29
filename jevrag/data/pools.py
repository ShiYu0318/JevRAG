"""Frozen candidate pools for reranking: every method ranks the same top-n."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from .drcd import Paragraph, Question

Retriever = Callable[[Sequence[str]], List[List[Tuple[str, float]]]]


def build_pools(questions: List[Question], retrieve: Retriever, depth: int = 20) -> List[dict]:
    ranked = retrieve([q.question for q in questions])
    out = []
    for q, run in zip(questions, ranked):
        out.append({
            "qid": q.qid,
            "question": q.question,
            "answers": q.answers,
            "qrels": {q.pid: 3},
            "candidates": [{"pid": pid, "rank": r, "score": round(s, 4)} for r, (pid, s) in enumerate(run[:depth], 1)],
        })
    return out


def load_pools(path: str | Path, limit: int | None = None) -> List[dict]:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    return rows[:limit] if limit else rows


def texts(pool: dict, paras: Dict[str, Paragraph]) -> List[str]:
    return [paras[c["pid"]].text for c in pool["candidates"]]
