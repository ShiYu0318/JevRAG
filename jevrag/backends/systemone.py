"""Client for any server speaking the ``POST /v1/systemone`` protocol.

Jev (via OpenRouter), Kev (``python -m kev.serve``) and Laya (``laya-serve``)
all accept the same body::

    {"model": ..., "state": ..., "questions": {key: {"type": ..., ...}}}

so a single client with a few presets covers all of them. Standard library only.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..cache import CallCache, cache_key


@dataclass
class Answer:
    """One typed answer, normalised across backends."""

    type: str
    raw: Dict[str, Any]

    @property
    def p(self) -> float:
        """P(yes) for a noul; confidence otherwise."""
        if self.type == "noul":
            return float(self.raw["noul"])
        return float(self.raw.get("confidence", max(self.probs.values(), default=0.0)))

    @property
    def label(self) -> Any:
        if self.type == "noul":
            return self.p >= 0.5
        if self.type == "choice":
            return self.raw.get("choice") or max(self.probs, key=self.probs.get)
        return self.raw.get("score", self.expected_score)

    @property
    def probs(self) -> Dict[str, float]:
        if self.type == "noul":
            return {"yes": self.p, "no": 1.0 - self.p}
        return {str(k): float(v) for k, v in self.raw.get("probabilities", {}).items()}

    @property
    def expected_score(self) -> float:
        """Expectation over score levels 0..n-1, a continuous relevance."""
        if self.type != "score":
            raise TypeError("expected_score only applies to score questions")
        return sum(int(k) * v for k, v in self.probs.items())


@dataclass
class Response:
    answers: Dict[str, Answer]
    latency_ms: float
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    cache_hit: bool = False

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]


class SystemOneError(RuntimeError):
    pass


class BudgetExceeded(SystemOneError):
    pass


@dataclass
class Meter:
    """Running totals for one client; read by the run manifest."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0
    versions: Dict[str, int] = field(default_factory=dict)


class SystemOneClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        path: str = "/v1/systemone",
        timeout: float = 60.0,
        retries: int = 2,
        extra_body: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        usd_per_mtok: float = 0.0,
        cache: Optional[CallCache] = None,
        pin_version: Optional[str] = None,
        budget_usd: Optional[float] = None,
        on_call: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.url = base_url.rstrip("/") + path
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.extra_body = extra_body or {}
        self.name = name or model
        self.usd_per_mtok = usd_per_mtok
        self.cache = cache
        self.pin_version = pin_version
        self.budget_usd = budget_usd
        self.on_call = on_call
        self.meter = Meter()
        self._lock = threading.Lock()

    # ---- calls ---------------------------------------------------------
    def ask(self, state: Any, questions: Dict[str, Dict[str, Any]], use_cache: bool = True,
            tag: Optional[Dict[str, Any]] = None) -> Response:
        body = {"model": self.model, "state": state, "questions": questions, **self.extra_body}
        key = cache_key(self.name, self.model, body)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key, self.pin_version)
            if hit is not None:
                payload, version = hit
                resp = self._parse(payload, payload.get("_latency_ms", 0.0), cache_hit=True)
                self._record(resp, tag)
                return resp

        if self.budget_usd is not None and self.meter.cost_usd >= self.budget_usd:
            raise BudgetExceeded(f"{self.name}: budget ${self.budget_usd} reached")

        payload, ms = self._post(body)
        payload["_latency_ms"] = ms
        resp = self._parse(payload, ms, cache_hit=False)
        if self.pin_version and resp.model != self.pin_version:
            raise SystemOneError(f"{self.name}: server returned {resp.model!r}, pinned {self.pin_version!r}")
        if self.cache is not None:
            self.cache.put(key, self.name, self.model, resp.model, body, payload)
        self._record(resp, tag)
        return resp

    def ask_many(self, jobs: List[tuple], workers: int = 8, **kw: Any) -> List[Response]:
        """Run several (state, questions) requests concurrently, order preserved."""
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda j: self.ask(*j, **kw), jobs))

    def server_info(self) -> Dict[str, Any]:
        """What the server reports about itself: GET /health (Laya reports checkpoint
        revisions there), else GET /v1/models (Kev reports the loaded run there)."""
        base = self.url.rsplit("/v1/", 1)[0]
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(base + path, timeout=5) as r:
                    return json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                continue
        return {}

    # ---- internals -----------------------------------------------------
    def _post(self, body: Dict[str, Any]) -> tuple:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                if "answers" not in payload:
                    raise KeyError("answers")
                return payload, (time.perf_counter() - t0) * 1000
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last = SystemOneError(f"{self.name}: HTTP {e.code}: {detail}")
                if e.code < 500 and e.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = SystemOneError(f"{self.name}: {e!r}")
            time.sleep(0.5 * (2 ** attempt))
        raise last  # type: ignore[misc]

    def _parse(self, payload: Dict[str, Any], ms: float, cache_hit: bool) -> Response:
        answers = {k: Answer(v["type"], v) for k, v in payload["answers"].items()}
        return Response(answers, ms, payload.get("usage", {}) or {}, payload.get("model", self.model), cache_hit)

    def _record(self, resp: Response, tag: Optional[Dict[str, Any]]) -> None:
        tokens = int(resp.usage.get("input_tokens") or resp.usage.get("prompt_tokens") or 0)
        cost = resp.usage.get("cost")
        cost = float(cost) if cost is not None else tokens * self.usd_per_mtok / 1e6
        with self._lock:
            m = self.meter
            m.calls += 1
            m.versions[resp.model] = m.versions.get(resp.model, 0) + 1
            if resp.cache_hit:
                m.cache_hits += 1
            else:
                m.input_tokens += tokens
                m.cost_usd += cost
        if self.on_call is not None:
            self.on_call({
                **(tag or {}),
                "backend": self.name,
                "model_version": resp.model,
                "answers": {k: a.raw for k, a in resp.answers.items()},
                "latency_ms": round(resp.latency_ms, 1),
                "usage": {**resp.usage, "input_tokens": tokens, "cost_usd": 0.0 if resp.cache_hit else cost},
                "cache_hit": resp.cache_hit,
            })


# ---- question builders ---------------------------------------------------
def noul(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: List[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}
