"""Reader for built benchmark files (data/build/<name>/<split>[.zh-Hans].jsonl)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional


def path_for(root: str | Path, split: str, script: str = "zh-Hant") -> Path:
    suffix = "" if script == "zh-Hant" else f".{script}"
    return Path(root) / f"{split}{suffix}.jsonl"


def load(root: str | Path, split: str, conditions: Optional[Iterable[str]] = None,
         script: str = "zh-Hant", limit: Optional[int] = None, seed: int = 7) -> List[dict]:
    conds = set(conditions) if conditions else None
    rows = []
    with path_for(root, split, script).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if conds is None or r["condition"] in conds:
                rows.append(r)
    if limit is not None and limit < len(rows):
        rows = sorted(random.Random(seed).sample(rows, limit), key=lambda r: r["id"])
    return rows


def passages(inst: dict) -> List[str]:
    return [c["text"] for c in inst["contexts"]]
