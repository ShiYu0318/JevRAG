from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def write_rows(path: str | Path, rows: Iterable[dict]) -> str:
    """Write JSONL and return the sha256 of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()
