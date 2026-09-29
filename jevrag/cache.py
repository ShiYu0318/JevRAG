"""On-disk cache for model calls.

Key = sha256(backend, requested model, canonical request JSON). The version
string reported by the server is stored with the payload, so a run can refuse
hits that were produced by a different model version.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(backend: str, model: str, request: Any) -> str:
    return hashlib.sha256(canonical([backend, model, request]).encode("utf-8")).hexdigest()


class CallCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            " key TEXT PRIMARY KEY, backend TEXT, model TEXT, version TEXT,"
            " request TEXT, response TEXT, created REAL)"
        )
        self._db.commit()

    def get(self, key: str, version: Optional[str] = None) -> Optional[Tuple[Dict[str, Any], str]]:
        with self._lock:
            row = self._db.execute("SELECT response, version FROM calls WHERE key = ?", (key,)).fetchone()
        if row is None or (version is not None and row[1] != version):
            return None
        return json.loads(row[0]), row[1]

    def put(self, key: str, backend: str, model: str, version: str, request: Any,
            response: Dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, backend, model, version, canonical(request), canonical(response), time.time()),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()
