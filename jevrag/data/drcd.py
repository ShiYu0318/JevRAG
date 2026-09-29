"""DRCD loader. Source: https://github.com/DRCKnowledgeTeam/DRCD (CC BY-SA 3.0)."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

SPLITS = ("training", "dev", "test")
URL = "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/DRCD_{split}.json"


@dataclass
class Paragraph:
    pid: str
    article: str
    title: str
    split: str
    text: str


@dataclass
class Question:
    qid: str
    pid: str
    article: str
    split: str
    question: str
    answers: List[str]
    answer_start: int


def _id(x: str) -> str:
    return x.replace("-", "_")


def download(raw_dir: str | Path, splits: Iterable[str] = SPLITS) -> None:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for s in splits:
        dst = raw / f"DRCD_{s}.json"
        if not dst.exists():
            urllib.request.urlretrieve(URL.format(split=s), dst)


def load(raw_dir: str | Path, splits: Iterable[str] = SPLITS):
    """Return (paragraphs by pid, questions by split)."""
    paras: Dict[str, Paragraph] = {}
    questions: Dict[str, List[Question]] = {}
    for s in splits:
        data = json.loads((Path(raw_dir) / f"DRCD_{s}.json").read_text(encoding="utf-8"))
        qs = []
        for art in data["data"]:
            for p in art["paragraphs"]:
                pid = _id(p["id"])
                paras[pid] = Paragraph(pid, str(art["id"]), art["title"], s, p["context"])
                for qa in p["qas"]:
                    texts = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                    qs.append(Question(_id(qa["id"]), pid, str(art["id"]), s, qa["question"], texts,
                                       int(qa["answers"][0]["answer_start"])))
        questions[s] = qs
    return paras, questions
