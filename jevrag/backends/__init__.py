"""Backend registry.

Every backend is reached through the same ``/v1/systemone`` client; only the
URL, model id and a couple of options differ. Ports and model ids can be
overridden with environment variables.
"""
from __future__ import annotations

import os
from typing import Any, List

from .systemone import (Answer, BudgetExceeded, Response, SystemOneClient, SystemOneError,
                        choice, noul, score)

JEV_USD_PER_MTOK = 0.042


def _jev(**kw: Any) -> SystemOneClient:
    key = kw.pop("api_key", None) or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemOneError("set OPENROUTER_API_KEY to use Jev via OpenRouter")
    model = os.environ.get("JEV_MODEL", "jev-1.13")
    return SystemOneClient("https://openrouter.ai/api", model, api_key=key, name="jev",
                           usd_per_mtok=JEV_USD_PER_MTOK, **kw)


def _kev(name: str) -> Any:
    # One Kev run per server process; the size is chosen at `kev.serve --run`.
    def make(**kw: Any) -> SystemOneClient:
        port = int(os.environ.get("KEV_PORT", 8009))
        return SystemOneClient(f"http://127.0.0.1:{port}", "kev-latest", name=name, **kw)
    return make


def _laya(checkpoint: str, name: str) -> Any:
    # Chinese text needs the multilingual checkpoint; the English one collapses
    # on non-Latin scripts while staying confident (kept only as a negative control).
    def make(**kw: Any) -> SystemOneClient:
        port = int(os.environ.get("LAYA_PORT", 8000))
        max_len = os.environ.get("LAYA_MAX_LEN")
        extra = {"max_len": int(max_len)} if max_len else {}
        return SystemOneClient(f"http://127.0.0.1:{port}", checkpoint, extra_body=extra, name=name, **kw)
    return make


def _mock(**kw: Any) -> SystemOneClient:
    port = int(os.environ.get("MOCK_PORT", 8765))
    return SystemOneClient(f"http://127.0.0.1:{port}", "mock", name="mock", **kw)


REGISTRY = {
    "jev": _jev,
    "kev4b": _kev("kev4b"),
    "kev08b": _kev("kev08b"),
    "kev27b": _kev("kev27b"),
    "laya-ml": _laya("multilingual", "laya-ml"),
    "laya-en": _laya("english", "laya-en"),
    "mock": _mock,
}
ALIASES = {"kev": "kev4b", "laya": "laya-ml"}


def get_backend(name: str, **kw: Any) -> SystemOneClient:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in REGISTRY:
        raise SystemOneError(f"unknown backend {name!r} ({'|'.join(REGISTRY)})")
    return REGISTRY[key](**kw)


def available() -> List[str]:
    return list(REGISTRY)


__all__ = ["Answer", "BudgetExceeded", "Response", "SystemOneClient", "SystemOneError",
           "choice", "noul", "score", "get_backend", "available", "REGISTRY"]
