"""A fake /v1/systemone server for testing the plumbing without any model.

Probabilities come from character-bigram overlap between the question and the
rest of the state, so outputs are deterministic and roughly sensible. It is not
a baseline.
"""
from __future__ import annotations

import json
import math
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def bigrams(s: str) -> set:
    s = "".join(ch for ch in s.lower() if not ch.isspace())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def flatten(x: Any) -> str:
    if isinstance(x, dict):
        return " ".join(flatten(v) for v in x.values())
    if isinstance(x, list):
        return " ".join(flatten(v) for v in x)
    return str(x)


def overlap(state: Any) -> float:
    if isinstance(state, dict) and "question" in state:
        q = bigrams(str(state["question"]))
        rest = bigrams(flatten({k: v for k, v in state.items() if k != "question"}))
    else:
        text = flatten(state)
        q, rest = bigrams(text[:40]), bigrams(text)
    return len(q & rest) / max(len(q), 1)


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def answer(body: dict) -> dict:
    base = overlap(body.get("state", ""))
    answers = {}
    for key, q in body["questions"].items():
        p = sigmoid(8 * (base - 0.35) + (zlib.crc32(key.encode()) % 7 - 3) * 0.05)
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": round(p, 4)}
            continue
        opts = list(q["criteria"]) if q["type"] == "choice" else list(range(len(q["criteria"])))
        n = len(opts)
        raw = [math.exp(-abs(i / max(n - 1, 1) - p) * 4) for i in range(n)]
        z = sum(raw)
        probs = {str(o): round(r / z, 4) for o, r in zip(opts, raw)}
        best = max(probs, key=probs.get)
        ans = {"type": q["type"], "probabilities": probs, "confidence": probs[best]}
        if q["type"] == "choice":
            ans["choice"] = best
        else:
            ans["score"] = round(sum(int(k) * v for k, v in probs.items()), 3)
        answers[key] = ans
    return {"model": "mock-0", "answers": answers,
            "usage": {"input_tokens": len(flatten(body.get("state", "")))}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        out = json.dumps({"status": "ok", "model": "mock-0"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        out = json.dumps(answer(body)).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def serve(port: int = 8765, background: bool = False) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        srv.serve_forever()
    return srv
