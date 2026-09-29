"""Run directories: calls.jsonl, manifest.json and the test-read log."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends.systemone import SystemOneClient
from .cache import canonical


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Run:
    """One experiment run. Use as a context manager so the manifest is always written."""

    def __init__(self, exp: str, backend: str, split: str, config: Dict[str, Any],
                 root: str | Path = "results", datasets: Optional[Dict[str, str]] = None) -> None:
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{exp}-{backend}-{split}-{stamp}"
        self.root = Path(root)
        self.dir = self.root / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.exp, self.backend, self.split = exp, backend, split
        self.config = config
        self.datasets = datasets or {}
        self.clients: List[SystemOneClient] = []
        self._calls = (self.dir / "calls.jsonl").open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self.started = now()
        self.summary: Dict[str, Any] = {}
        self.server_info: Dict[str, Any] = {}

    def log_call(self, row: Dict[str, Any]) -> None:
        line = json.dumps({"run_id": self.id, **row}, ensure_ascii=False)
        with self._lock:
            self._calls.write(line + "\n")

    def attach(self, client: SystemOneClient) -> SystemOneClient:
        client.on_call = self.log_call
        self.clients.append(client)
        if "openrouter" not in client.url:
            self.server_info[client.name] = client.server_info()
        return client

    def write(self, name: str, obj: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def manifest(self) -> Dict[str, Any]:
        return {
            "run_id": self.id,
            "experiment": self.exp,
            "backend": self.backend,
            "split": self.split,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "config_sha256": hashlib.sha256(canonical(self.config).encode("utf-8")).hexdigest(),
            "config": self.config,
            "datasets": self.datasets,
            "model_versions": {c.name: c.meter.versions for c in self.clients},
            "server_info": self.server_info,
            "calls": sum(c.meter.calls for c in self.clients),
            "cache_hits": sum(c.meter.cache_hits for c in self.clients),
            "input_tokens": sum(c.meter.input_tokens for c in self.clients),
            "cost_usd": round(sum(c.meter.cost_usd for c in self.clients), 6),
            "started": self.started,
            "finished": now(),
            "python": platform.python_version(),
            "test_read": self.split == "test",
            "summary": self.summary,
        }

    def close(self) -> None:
        self._calls.close()
        m = self.manifest()
        self.write("manifest.json", m)
        if m["test_read"]:
            log = self.root / "TEST_LOG.md"
            new = not log.exists()
            with log.open("a", encoding="utf-8") as f:
                if new:
                    f.write("| run | experiment | backend | commit | finished |\n|---|---|---|---|---|\n")
                f.write(f"| {self.id} | {self.exp} | {self.backend} | {m['git_commit'][:10]} | {m['finished']} |\n")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
