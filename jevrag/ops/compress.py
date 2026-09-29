"""Extractive compression: one keep/drop noul per sentence."""
from __future__ import annotations

import re
from typing import List

from ..backends.systemone import SystemOneClient
from ..questions import keep_v1

_SENT = re.compile(r"(?<=[。！？!?；;])")


def split_sentences(text: str) -> List[str]:
    return [s for s in (x.strip() for x in _SENT.split(text)) if s]


def compress(client: SystemOneClient, query: str, passage: str, tau: float = 0.5,
             window: bool = True, keep_prev: bool = False) -> str:
    sents = split_sentences(passage)
    jobs = [keep_v1(query, s,
                    before=sents[i - 1] if window and i > 0 else "",
                    after=sents[i + 1] if window and i + 1 < len(sents) else "")
            for i, s in enumerate(sents)]
    keep = [r["keep"].p >= tau for r in client.ask_many(jobs)]
    if keep_prev:
        keep = [k or (i + 1 < len(keep) and keep[i + 1]) for i, k in enumerate(keep)]
    return "".join(s for s, k in zip(sents, keep) if k)
