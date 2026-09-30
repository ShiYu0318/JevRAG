"""Local demo console: experiment results, milestones and a live judge.

    python -m jevrag demo --port 8900

Binds to 127.0.0.1 only. Everything it shows comes from local files
(results/, the built benchmark, an optional milestones file).
"""
from __future__ import annotations

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ..backends import SystemOneClient, SystemOneError, get_backend, run_mismatch
from ..cache import CallCache
from ..ops.sufficiency import aggregate, decide_action
from ..questions import grade_v1, sufficiency_v1

STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
MAX_PASSAGES = 10
MAX_CHARS = 4000


def latest_runs(results: Path) -> Dict[str, Dict[str, Any]]:
    """{experiment: {"backend / split": {manifest subset, result}}} for the newest run of each."""
    out: Dict[str, Dict[str, Any]] = {}
    for man_path in sorted(results.glob("E*-*/manifest.json")):
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        res_path = man_path.parent / f"{man['experiment']}.json"
        if not res_path.exists():
            continue  # still running or failed
        key = f"{man['backend']} / {man['split']}"
        out.setdefault(man["experiment"], {})[key] = {
            "run_id": man["run_id"], "finished": man.get("finished"), "calls": man.get("calls"),
            "cache_hits": man.get("cache_hits"), "cost_usd": man.get("cost_usd"),
            "model_versions": man.get("model_versions"), "config": man.get("config", {}),
            "result": json.loads(res_path.read_text(encoding="utf-8")),
        }
    return out


def running_runs(results: Path) -> List[Dict[str, Any]]:
    """Run directories that have a call log but no manifest yet."""
    out = []
    for d in sorted(results.glob("E*-*")):
        calls = d / "calls.jsonl"
        if d.is_dir() and calls.exists() and not (d / "manifest.json").exists():
            with calls.open(encoding="utf-8") as f:
                n = sum(1 for _ in f)
            out.append({"run_id": d.name, "calls": n, "updated": calls.stat().st_mtime})
    return out


class Benchmark:
    """Lazy, cached view of one built benchmark directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._rows: Dict[str, List[dict]] = {}
        self._lock = threading.Lock()

    def rows(self, split: str) -> List[dict]:
        with self._lock:
            if split not in self._rows:
                path = self.root / f"{split}.jsonl"
                self._rows[split] = ([json.loads(l) for l in path.open(encoding="utf-8")]
                                     if path.exists() else [])
            return self._rows[split]

    def sample(self, split: str, condition: Optional[str], n: int, seed: Optional[int]) -> List[dict]:
        rows = [r for r in self.rows(split) if not condition or r["condition"] == condition]
        rng = random.Random(seed)
        return rng.sample(rows, min(n, len(rows)))

    def get(self, split: str, instance_id: str) -> Optional[dict]:
        return next((r for r in self.rows(split) if r["id"] == instance_id), None)


class Judge:
    """Runs the decision points for one question on several backends."""

    def __init__(self, cache: Optional[CallCache]) -> None:
        self.cache = cache
        self.clients: Dict[str, SystemOneClient] = {}

    def client(self, name: str) -> SystemOneClient:
        if name not in self.clients:
            self.clients[name] = get_backend(name, cache=self.cache, timeout=30.0, retries=1)
        return self.clients[name]

    def available(self, names: List[str]) -> Dict[str, Any]:
        out = {}
        for name in names:
            try:
                c = self.client(name)
            except SystemOneError as e:
                out[name] = {"ok": False, "reason": str(e)}
                continue
            if "openrouter" in c.url:
                out[name] = {"ok": True, "remote": True}
            else:
                info = c.server_info()
                reason = run_mismatch(c, info) if info else f"no server at {c.url.rsplit('/v1/', 1)[0]}"
                out[name] = {"ok": not reason, "remote": False, "reason": reason}
        return out

    def run(self, name: str, question: str, passages: List[str], lang: str = "en") -> Dict[str, Any]:
        c = self.client(name)
        before = c.meter.cost_usd
        t0 = time.perf_counter()
        grades = c.ask_many([grade_v1(question, p, lang=lang, safety=True) for p in passages], workers=8)
        verdict = c.ask(*sufficiency_v1(question, passages, lang=lang))
        ms = (time.perf_counter() - t0) * 1000
        ans = [g["ans"].p for g in grades]
        return {
            "backend": name,
            "model": verdict.model,
            "passages": [{"answers": g["ans"].p, "relevance": g["rel"].expected_score,
                          "injection": g["inj"].p, "contradicts": g["con"].p} for g in grades],
            "verdict": verdict["verdict"].probs,
            "false_premise": verdict["false_premise"].p,
            "evidence_noisy_or": aggregate(ans, "noisy_or"),
            "action": decide_action(verdict),
            "latency_ms": round(ms),
            "cached": all(g.cache_hit for g in grades) and verdict.cache_hit,
            "cost_usd": c.meter.cost_usd - before,
        }


def make_handler(results: Path, bench: Benchmark, judge: Judge, milestones: Optional[Path],
                 backends: List[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def _json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents or not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("content-type", TYPES.get(path.suffix, "application/octet-stream"))
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/results":
                return self._json({"runs": latest_runs(results), "running": running_runs(results)})
            if url.path == "/api/milestones":
                data = json.loads(milestones.read_text(encoding="utf-8")) if milestones and milestones.exists() else []
                return self._json(data)
            if url.path == "/api/backends":
                return self._json(judge.available(backends))
            if url.path == "/api/instances":
                seed = int(q["seed"]) if q.get("seed") else None
                rows = bench.sample(q.get("split", "dev"), q.get("condition") or None, int(q.get("n", 1)), seed)
                return self._json(rows)
            return self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/judge":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            except ValueError:
                return self._json({"error": "request body is not JSON"}, 400)
            question = str(body.get("question", "")).strip()
            passages = [str(p)[:MAX_CHARS] for p in body.get("passages", []) if str(p).strip()][:MAX_PASSAGES]
            names = [b for b in body.get("backends", []) if b in backends]
            if not question or not passages or not names:
                return self._json({"error": "send a question, at least one passage and one backend"}, 400)
            out = []
            for name in names:
                try:
                    out.append(judge.run(name, question, passages, body.get("lang", "en")))
                except SystemOneError as e:
                    out.append({"backend": name, "error": str(e)})
            return self._json(out)

    return Handler


def serve(port: int = 8900, results: str | Path = "results", dataset: str | Path = "data/build/jevrag-tc-v0.1.0-k3",
          milestones: Optional[str | Path] = "docs/milestones.json", cache: Optional[str | Path] = "cache/calls.sqlite",
          backends: Optional[List[str]] = None, background: bool = False) -> ThreadingHTTPServer:
    handler = make_handler(Path(results), Benchmark(Path(dataset)), Judge(CallCache(cache) if cache else None),
                           Path(milestones) if milestones else None,
                           backends or ["jev", "laya-ml", "laya-en", "kev08b", "kev4b", "mock"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        print(f"demo on http://127.0.0.1:{port}")
        srv.serve_forever()
    return srv
